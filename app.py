
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session
)
import os
import json
import pandas as pd
import traceback
from driver import main as run_pipeline
from auth import init_db, register_user, validate_user

app = Flask(__name__)
app.secret_key = "YOUR-SECRET-KEY"  # Replace with a secure key in production

# Initialize the user database
init_db()

def assign_sentiment(top_emotion, confidence):
    """Assign sentiment based on emotion and confidence."""
    if confidence < 0.5:
        return "Neutral"
    return "Negative" if top_emotion in ["Anger", "Sadness", "Fear"] else "Positive"

def ratio_as_percentage(part, whole):
    """Calculate part/whole as a percentage."""
    return round((part / whole) * 100, 2) if whole else 0.0

def login_required(f):
    """Decorator to enforce login for protected routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Handle user signup."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("signup.html", error="Please fill in all fields.")

        if register_user(username, password):
            return redirect(url_for("login"))
        return render_template("signup.html", error="Username is already taken.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if validate_user(username, password):
            session["username"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Log out the user and clear the session."""
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/", methods=["GET"])
@login_required
def index():
    """Render the main index page."""
    return render_template("index.html", username=session.get("username"))

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    """
    Process and analyze YouTube video data.
    Now the pipeline produces a final CSV (final_predictions.csv)
    that contains both emotion predictions and irony (sarcasm) outputs.
    """
    try:
        youtube_link = request.form.get("yt_link", "").strip()
        if not youtube_link:
            return jsonify({"error": "No YouTube link provided"}), 400

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline(youtube_link, output_dir, cleanup=False)

        # Instead of emotion_predictions.csv, we now load final_predictions.csv.
        final_csv_path = os.path.join(output_dir, "final_predictions.csv")
        if not os.path.exists(final_csv_path):
            return jsonify({"error": "final_predictions.csv not found"}), 500

        df_final = pd.read_csv(final_csv_path)
        required_cols = ["comment", "comment_emotion1", "comment_conf1", "Irony", "Irony Probability"]
        if not all(col in df_final.columns for col in required_cols):
            return jsonify(
                {"error": f"Missing columns: {required_cols} in final_predictions.csv"}
            ), 500

        # Assign sentiment for each comment based on emotion predictions
        df_final["sentiment"] = df_final.apply(
            lambda row: assign_sentiment(row["comment_emotion1"], row["comment_conf1"]),
            axis=1
        )
        report_data = df_final.to_dict(orient="records")

        # Determine overall sentiment and emotion for all comments
        overall_sentiment = "N/A"
        overall_emotion = "N/A"
        if not df_final.empty:
            sentiment_counts = df_final["sentiment"].value_counts()
            if not sentiment_counts.empty:
                overall_sentiment = sentiment_counts.idxmax()

            emotion_counts = df_final["comment_emotion1"].value_counts()
            if not emotion_counts.empty:
                overall_emotion = emotion_counts.idxmax()

        # Video metadata & engagement metrics
        video_json_path = os.path.join(output_dir, "video_data.json")
        title, channel_name, upload_date, tags = "", "", "", []
        likes_to_views_perc = comments_to_views_perc = views_to_subs_perc = 0.0

        if os.path.exists(video_json_path):
            with open(video_json_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            title = video_data.get("title", "No Title")
            channel_name = video_data.get("channel_name", "Unknown Channel")
            tags = video_data.get("tags", [])
            views = video_data.get("views", 0)
            likes = video_data.get("likes", 0)
            upload_date = video_data.get("upload_date", "Unknown Date")
            comment_count = video_data.get("comment_count", 0)
            subscriber_count = video_data.get("subscriber_count", 0)

            likes_to_views_perc = ratio_as_percentage(likes, views)
            comments_to_views_perc = ratio_as_percentage(comment_count, views)
            views_to_subs_perc = ratio_as_percentage(views, subscriber_count)
        else:
            print("video_data.json not found. Skipping engagement metrics.")

        sentiment_counts = df_final["sentiment"].value_counts().to_dict()
        emotion_counts = df_final["comment_emotion1"].value_counts().to_dict()
        irony_counts = df_final["Irony"].value_counts().to_dict()
        # ------------------------------------------------------------
        # 📊 TRANSCRIPT-LEVEL EMOTION DISTRIBUTION
        # ------------------------------------------------------------
        chunks_csv_path = os.path.join(output_dir, "chunks.csv")
        transcript_emotion_counts = {}
        if os.path.exists(chunks_csv_path):
            df_chunks = pd.read_csv(chunks_csv_path)
            if "transcript_emotion1" in df_chunks.columns:
                transcript_emotion_counts = (
                    df_chunks["transcript_emotion1"]
                    .value_counts()
                    .to_dict()
                )
        else:
            print("⚠️ chunks.csv not found – transcript distribution will be empty")


        # Pass along the Irony columns for visualization
        return render_template(
            "results.html",
            video_link=youtube_link,
            report_data=report_data,
            overall_sentiment=overall_sentiment,
            overall_emotion=overall_emotion,
            video_title=title,
            channel_name=channel_name,
            tags=tags,
            upload_date=upload_date,
            total_views=views,
            total_likes=likes,
            total_comments=comment_count,
            total_subscribers=subscriber_count,
            likes_to_views_perc=likes_to_views_perc,
            comments_to_views_perc=comments_to_views_perc,
            views_to_subs_perc=views_to_subs_perc,
            sentiment_distribution=sentiment_counts,
            emotion_distribution=emotion_counts,
            irony_distribution=irony_counts,
            transcript_emotion_distribution=transcript_emotion_counts,
            # Additional fields for irony visualization:
            # Each record in report_data will include "Irony" and "Irony Probability"
            username=session.get("username")
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

'''

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    send_file
)
import os
import json
import pandas as pd
import traceback
from driver import main as run_pipeline
from auth import init_db, register_user, validate_user

app = Flask(__name__)
app.secret_key = "YOUR-SECRET-KEY"  # Replace with a secure key in production

# -------------------------
#  DB init & helpers
# -------------------------
init_db()

def assign_sentiment(emotion: str, conf: float) -> str:
    """Crude sentiment from top emotion & confidence."""
    if conf < 0.5:
        return "Neutral"
    return "Negative" if emotion in {"Anger", "Sadness", "Fear"} else "Positive"


def ratio_as_percentage(part: float, whole: float) -> float:
    return round((part / whole) * 100, 2) if whole else 0.0


# -------------------------
#  Auth decorators
# -------------------------
from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# -------------------------
#  Routes – auth
# -------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "").strip()
        if not u or not p:
            return render_template("signup.html", error="Please fill in all fields.")
        if register_user(u, p):
            return redirect(url_for("login"))
        return render_template("signup.html", error="Username is already taken.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "").strip()
        if validate_user(u, p):
            session["username"] = u
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

# -------------------------
#  Core pages
# -------------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session.get("username"))

# -------------------------
#  Analysis pipeline
# -------------------------
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    try:
        youtube_link = request.form.get("yt_link", "").strip()
        if not youtube_link:
            return jsonify({"error": "No YouTube link provided"}), 400

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline(youtube_link, output_dir, cleanup=False)

        final_csv = os.path.join(output_dir, "final_predictions.csv")
        if not os.path.exists(final_csv):
            return jsonify({"error": "final_predictions.csv not found"}), 500

        df = pd.read_csv(final_csv)
        needed = {"comment", "comment_emotion1", "comment_conf1", "Irony", "Irony Probability"}
        if not needed.issubset(df.columns):
            return jsonify({"error": f"final_predictions.csv missing {needed}"}), 500

        # Sentiment
        df["sentiment"] = df.apply(lambda r: assign_sentiment(r["comment_emotion1"], r["comment_conf1"]), axis=1)

        # Tone KPI numbers (if columns present)
        if {"ToneVariationScore", "ToneVariationLabel"}.issubset(df.columns):
            avg_tone = round(df["ToneVariationScore"].mean(), 3)
            percent_likely = round(100 * (df["ToneVariationLabel"] == "Likely Tone-Shift").mean(), 1)
            tone_dist = df["ToneVariationLabel"].value_counts().to_dict()
        else:
            avg_tone = percent_likely = 0
            tone_dist = {}

        report_data = df.to_dict(orient="records")

        # Aggregate sentiment & emotion
        overall_sent = df["sentiment"].mode()[0] if not df.empty else "N/A"
        overall_emot = df["comment_emotion1"].mode()[0] if not df.empty else "N/A"

        # Video metadata ------------------------------------------------
        meta_path = os.path.join(output_dir, "video_data.json")
        views = likes = comment_count = subscriber_count = 0
        title = channel_name = upload_date = ""
        tags = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            title = meta.get("title", "No Title")
            channel_name = meta.get("channel_name", "Unknown")
            tags = meta.get("tags", [])
            views = meta.get("views", 0)
            likes = meta.get("likes", 0)
            upload_date = meta.get("upload_date", "Unknown Date")
            comment_count = meta.get("comment_count", 0)
            subscriber_count = meta.get("subscriber_count", 0)

        sentiment_dist = df["sentiment"].value_counts().to_dict()
        emotion_dist = df["comment_emotion1"].value_counts().to_dict()

        return render_template(
            "results.html",
            # link & table
            video_link=youtube_link,
            report_data=report_data,
            # KPI
            avg_tone_score=avg_tone,
            percent_likely=percent_likely,
            tone_distribution=tone_dist,
            # overall
            overall_sentiment=overall_sent,
            overall_emotion=overall_emot,
            # meta
            video_title=title,
            channel_name=channel_name,
            tags=tags,
            upload_date=upload_date,
            total_views=views,
            total_likes=likes,
            total_comments=comment_count,
            total_subscribers=subscriber_count,
            likes_to_views_perc=ratio_as_percentage(likes, views),
            comments_to_views_perc=ratio_as_percentage(comment_count, views),
            views_to_subs_perc=ratio_as_percentage(views, subscriber_count),
            sentiment_distribution=sentiment_dist,
            emotion_distribution=emotion_dist,
            username=session.get("username"),
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------
#  Download filtered CSV
# -------------------------
@app.route("/download_filtered")
@login_required
def download_filtered():
    level = request.args.get("level", "Likely Tone-Shift")
    output_dir = "output"
    final_csv = os.path.join(output_dir, "final_predictions.csv")
    if not os.path.exists(final_csv):
        return "File not ready", 404
    df = pd.read_csv(final_csv)
    if "ToneVariationLabel" not in df.columns:
        return "ToneVariationLabel column missing", 400
    sub = df[df["ToneVariationLabel"] == level]
    tmp = os.path.join(output_dir, f"filtered_{level.replace(' ', '_')}.csv")
    sub.to_csv(tmp, index=False)
    return send_file(tmp, as_attachment=True)

# -------------------------
#  Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''