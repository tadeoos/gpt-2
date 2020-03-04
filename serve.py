from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse
import gpt_2_simple as gpt2
import tensorflow as tf
import uvicorn
import os
import gc

middleware = [
    Middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=["*"])
]

app = Starlette(debug=False,
                middleware=middleware
                )

sess = gpt2.start_tf_sess(threads=1)
gpt2.load_gpt2(sess, model_name=os.environ.get("MODEL_NAME", '1558M'), model_dir=os.environ.get("MODEL_DIR", '/models'))

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

generate_count = 0


@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess

    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return UJSONResponse({'text': ''}, headers=response_header)

    text = gpt2.generate(sess,
                         model_name=os.environ.get("MODEL_NAME", '1558M'),
                         model_dir=os.environ.get("MODEL_DIR", '/models'),
                         length=int(params.get('length', 1023)),
                         temperature=float(params.get('temperature', 0.7)),
                         top_k=int(params.get('top_k', 40)),
                         top_p=float(params.get('top_p', 1)),
                         prefix=params.get('prefix', ''),
                         truncate=params.get('truncate', '<|endoftext|>'),
                         include_prefix=str(params.get('include_prefix', False)).lower() == 'true',
                         return_as_list=True
                         )[0]

    generate_count += 1
    if generate_count == 8:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    # text = text[len('<|startoftext|>'):]
    return UJSONResponse({'text': text},
                         headers=response_header)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
