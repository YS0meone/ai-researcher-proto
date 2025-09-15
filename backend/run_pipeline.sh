docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf
uv run python -m app.data_pipeline -n 50 -q "cat:cs.CL AND all:nlp OR all:natural language processing" -o ./papers -p 2