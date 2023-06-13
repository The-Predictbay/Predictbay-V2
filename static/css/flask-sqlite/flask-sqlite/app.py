from flask import Flask, render_template, request, session, redirect, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = "your_secret_key"


@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('profile'))
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            error = 'Username already exists. Please choose a different username.'
            conn.close()
            return render_template('register.html', error=error)
        else:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            session['username'] = username
            return redirect(url_for('profile'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        if user and user[1] == password:
            conn.close()
            session['username'] = username
            return redirect(url_for('profile'))
        else:
            error = 'Invalid username or password.'
            conn.close()
            return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password=? WHERE username=?", (new_password, session['username']))
        conn.commit()
        conn.close()
        return redirect(url_for('logout'))

    return render_template('profile.html')


if __name__ == '__main__':
    app.run(debug=True)
