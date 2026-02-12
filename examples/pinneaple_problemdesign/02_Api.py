from flask import Flask, request, jsonify
from pinneaple_problemdesign import GeminiProvider, DesignAgent

app = Flask(__name__)

llm = GeminiProvider(model="gemini-2.0-flash")
agent = DesignAgent(llm=llm)

STATE = agent.start()

@app.post("/design/chat")
def design_chat():
    data = request.get_json(force=True)
    user_text = data.get("message", "")

    agent.ingest_user_message(STATE, user_text)
    out = agent.step(STATE)

    if out["type"] == "questions":
        return jsonify({
            "status": "needs_info",
            "stage": out["stage"],
            "questions_text": out["questions_text"],
            "questions": [g.__dict__ for g in out["questions"]],
            "warnings": out.get("warnings", []),
        })

    return jsonify({
        "status": "done",
        "markdown": out["markdown"],
        "spec": out["report"].spec.to_dict(),
        "gaps": [g.__dict__ for g in out["report"].gaps],
        "plan": {
            "recommended_approach": out["report"].plan.recommended_approach,
            "alternatives": out["report"].plan.alternatives,
            "steps": [
                {
                    "title": s.title,
                    "why": s.why,
                    "actions": s.actions,
                    "pinneaple_modules": s.pinneaple_modules,
                    "exit_criteria": s.exit_criteria,
                }
                for s in out["report"].plan.steps
            ],
            "go_no_go": out["report"].plan.go_no_go,
        }
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)
