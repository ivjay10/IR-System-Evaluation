from flask import Flask, render_template, request, send_from_directory
from ir_system import IRSystem
import os

app = Flask(__name__)

# Initialize IR System
current_dir = os.path.dirname(os.path.abspath(__file__))
documents_path = os.path.join(current_dir, "documents")
covers_path = os.path.join(current_dir, "covers")

ir_system = IRSystem(documents_path, covers_path)
ir_system.read_documents()
ir_system.build_inverted_index()
ir_system.compute_idf()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = ir_system.search(query)
        processed_results = []
        for doc_id, score in results[:10]:  # Display top 10 results
            title = doc_id.replace('.txt', '').replace('_', ' ').title()
            snippet = ir_system.documents[doc_id][:200] + "..."  # First 200 characters as snippet
            cover_path = ir_system.get_book_cover(doc_id)
            processed_results.append({
                'title': title,
                'score': f"{score:.4f}",
                'snippet': snippet,
                'cover': os.path.basename(cover_path)
            })
        return render_template('results.html', query=query, results=processed_results)
    return render_template('index.html')

@app.route('/covers/<path:filename>')
def serve_cover(filename):
    return send_from_directory(covers_path, filename)

if __name__ == '__main__':
    app.run(debug=True)