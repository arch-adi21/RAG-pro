# RAG-pro

### To copy my methods here are the simple steps you need to follow :-

1. Copy the code in app.py untill the endpoints . You need to copy the main function at the end , as the app initiation should already be done , if you are not dumb enough.
2. Install the requirements , you can install the requirements from requirements.txt file using `pip install -r requirements.txt`.
3. Initiate the app . If your app initiation is in app.py use `python app.py` or `python3 app.py`.
4. Use the following curl commands one by one (make sure you replace the dummy variables) :
   - ```bash
     curl -X POST -H "Content-Type: application/json" \
     -d '{"transcript_path": "path_to_the_transcript_pdf",\
     "pdf_paths": ["path_to_study_resource_book_pdf"]}' \
     http://127.0.0.1:5000/create_knowledge_base
     ```
   - ```bash
     curl -X POST http://localhost:5000/load_knowledge_base
     ```
   - ```bash
     curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"query": "What is your query ?"}' \
     http://localhost:5000/query
     ```
     
