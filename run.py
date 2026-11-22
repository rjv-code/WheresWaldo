from app import create_app

app = create_app()

# limit uploads to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10485760

if __name__ == "__main__":
    app.run(debug=True)
