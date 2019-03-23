from flask import Flask, jsonify, request
from flask_cors import CORS

from app import app, db, manager
from app.models.databases import CorpusChatbot

CORS(app, supports_credentials=True)

@app.route('/add_info', methods=["POST"])
def add_info():
        question = request.form['input']
        answer = request.form['output']

        question_in_table = CorpusChatbot.query.filter(CorpusChatbot.question==question).first()
        if question_in_table:
                return jsonify({"number":"250", "info":"question already exists in database, do not saved!"})
        else:
                corpus_chatbot = CorpusChatbot()
                corpus_chatbot.question = question
                corpus_chatbot.answer = answer
                db.session.add(corpus_chatbot)
                db.session.commit()
                return jsonify({"info":"saved data","saved_data":{"question":question, "answer":answer}})
if __name__ == "__main__":
        db.create_all()
        # manager.run()
        app.run(host='0.0.0.0', port=8090, debug=True)
