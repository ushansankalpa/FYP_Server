import base64
import csv
import os
import uuid

import jsonify as jsonify
import mysql.connector
import hashlib
import hmac
import psycopg2
from auth.auth_handler import signJWT


def registerUser(Userdata, learningstyle):
    mydb =  mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )

    mycursor = mydb.cursor()

    try:
        hashed_password = hashlib.sha256(Userdata.password.encode()).hexdigest()
        user = (Userdata.fullname, Userdata.email, hashed_password, Userdata.role, learningstyle)
        sql = "INSERT INTO user (fullname, email, password, role, learning_style) VALUES (%s, %s, %s, %s, %s)"
        mycursor.execute(sql, user)
        mydb.commit()
        # Get the ID of the newly inserted user
        # user_id = mycursor.lastrowid

        # Insert user measurements into usermeasurements table
        # measurements_data = (user_id, Userdata.name, Userdata.age, Userdata.gender,None,None,None,None,None,None,None,None,None,None,None,None)
        # measurements_sql = "INSERT INTO usermeasurements (userId, name,age, gender,neck,knee,ankle,biceps,forearm,wrist,weight, height, abdomen,chest,hip,thigh) VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s, %s)"
        # mycursor.execute(measurements_sql, measurements_data)
        # mydb.commit()
        print(mydb.commit())
        return user

    except mysql.connector.Error as error:
        print("Error while inserting data to MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return False

    finally:
        mydb.close()


def loginUser(userData):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )

    mycursor = mydb.cursor()

    try:
        hashed_password = hashlib.sha256(userData.password.encode()).hexdigest()
        query = "SELECT * FROM user WHERE email=%s AND password=%s"
        mycursor.execute(query, (userData.email, hashed_password))
        result = mycursor.fetchone()

        # if result:
        #
        #     return True
        # else:
        #
        #     return False

        if result is not None and result[1]:
            # Return user ID along with login success status
            return True
        else:
            return False

    except mysql.connector.Error as error:
        print("Error while querying data from MySQL: {}".format(error))
        return False

    finally:
        mydb.close()


def getUserDataByEmail(user_id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )

    mycursor = mydb.cursor()

    try:
        sql = "SELECT id, fullname, email, role, learning_style FROM user WHERE email = %s"
        mycursor.execute(sql, (user_id,))
        result = mycursor.fetchone()

        if result:
            user_data = {
                "id": result[0],
                "fullname": result[1],
                "email": result[2],
                "role": result[3],
                "learning_style": result[4]
            }

            return signJWT(user_data)
        else:
            return "No userData found!"

    except mysql.connector.Error as error:
        print("Error while retrieving data from MySQL: {}".format(error))
        return "No userData found!"

    finally:
        mydb.close()


def getUserDataById(user_id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )

    mycursor = mydb.cursor()

    try:
        sql = "SELECT id, fullname, email, role, learning_style FROM user WHERE id = %s"
        mycursor.execute(sql, (user_id,))
        result = mycursor.fetchone()

        if result:
            user_data = {
                "id": result[0],
                "fullname": result[1],
                "email": result[2],
                "role": result[3],
                "learning_style": result[4]
            }

            return user_data
        else:
            return "No userData found!"

    except mysql.connector.Error as error:
        print("Error while retrieving data from MySQL: {}".format(error))
        return "No userData found!"

    finally:
        mydb.close()


def get_all_users():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        # Execute a SELECT query to get all data rows from the resource table
        mycursor.execute("SELECT * FROM user")

        result = []
        for row in mycursor.fetchall():
            result.append({
                'id': row[0],
                'fullname': row[1],
                'email': row[2],
                'role': row[4],
                'learning_style': row[5]
            })

        return result

    except mysql.connector.Error as error:
        print("Error while updating data in MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return (False, "Unable to update")
    finally:
        mydb.close()

def get_learning_styles():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        # Execute a SELECT query to get all data rows from the resource table
        mycursor.execute("SELECT learning_style, COUNT(*) AS value FROM user GROUP BY learning_style;")

        result = []
        for row in mycursor.fetchall():
            result.append({
                'name': row[0],
                'value': row[1]
            })

        return result

    except mysql.connector.Error as error:
        print("Error while updating data in MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return (False, "Unable to update")
    finally:
        mydb.close()

def get_all_resources():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        # Execute a SELECT query to get all data rows from the resource table
        mycursor.execute("SELECT * FROM resource")
        os.makedirs("images", exist_ok=True)

        result = []
        for row in mycursor.fetchall():
            image_path = None
            if row[4] is not None:
                image_data = row[4]
                image_path = "data:image/jpeg;base64," + base64.b64encode(image_data).decode()
            result.append({
                'res_id': row[0],
                'title': row[1],
                'desc': row[2],
                'link': row[3],
                'image': image_path,
                'type': row[5],
                'field': row[6]
            })

        return result

    except mysql.connector.Error as error:
        print("Error while updating data in MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return (False, "Unable to update")
    finally:
        mydb.close()


def updateUserLearninStyle(learning_style, id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        sql = "UPDATE user SET learning_style = %s WHERE id = %s"
        val = (learning_style, id)
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record(s) affected")
        return (True, "succeffully updated")

    except mysql.connector.Error as error:
        print("Error while updating data in MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return (False, "Unable to update")

    finally:
        mydb.close()


def make_ratings(data, res_id, user_id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        query = "SELECT * FROM ratings WHERE res_id=%s AND user_id=%s"
        mycursor.execute(query, (res_id, user_id))
        result = mycursor.fetchone()
        if result:
            sql = "UPDATE ratings SET rate = %s WHERE res_id=%s AND user_id=%s"
            val = (data.rate, res_id, user_id)
            mycursor.execute(sql, val)
            mydb.commit()
            return (True, "succeffully updated")
        else:
            user_data = getUserDataById(user_id)
            ratings = (data.user_id, data.res_id, data.rate, user_data['learning_style'])
            sql = "INSERT INTO ratings (user_id, res_id, rate,  learning_style) VALUES (%s, %s, %s, %s)"
            mycursor.execute(sql, ratings)
            mydb.commit()
            return ratings

    except mysql.connector.Error as error:
        print("Error while inserting data to MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return False
    finally:
        mydb.close()


def export_csv():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )
    mycursor = mydb.cursor()

    try:
        query = "SELECT * FROM ratings;"

        # execute query and fetch data into a variable
        mycursor.execute(query)
        data = mycursor.fetchall()

        columns = [desc[0] for desc in mycursor.description]

        # write data to csv file with column names included
        with open("ratings.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            writer.writerows(data)

        return {"message": "Table exported as CSV file."}

    except mysql.connector.Error as error:
        print("Error while inserting data to MySQL: {}".format(error))
        try:
            mydb.rollback()
        except mysql.connector.Error as rollback_error:
            print("Error while rolling back changes to MySQL: {}".format(rollback_error))
        return False
    finally:
        mydb.close()


def getResDataByIds(res_id_list):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="learnMaster"
    )

    mycursor = mydb.cursor()

    try:
        res_ids = [item['res_id'] for item in res_id_list]
        placeholders = ', '.join(['%s'] * len(res_ids))
        sql = "SELECT res_id, title, `desc`, link, image, type, field FROM resource WHERE res_id IN ({})".format(
            placeholders)
        mycursor.execute(sql, tuple(res_ids))

        result = []
        for row in mycursor.fetchall():
            image_path = None
            if row[4] is not None:
                image_data = row[4]
                image_path = "data:image/jpeg;base64," + base64.b64encode(image_data).decode()
            result.append({
                'res_id': row[0],
                'title': row[1],
                'desc': row[2],
                'link': row[3],
                'image': image_path,
                'type': row[5],
                'field': row[6],
                'rating': next(item['rating'] for item in res_id_list if item['res_id'] == row[0])
            })

        return result

    #mycursor.executemany(sql, res_id)



        # for row in mycursor.fetchall():
        #     image_path = None
        #     if row[4] is not None:
        #         image_data = row[4]
        #         image_path = "data:image/jpeg;base64," + base64.b64encode(image_data).decode()
        #     result.append({
        #         'res_id': row[0],
        #         'title': row[1],
        #         'desc': row[2],
        #         'link': row[3],
        #         'image': image_path,
        #         'type': row[5],
        #         'field': row[6]
        #     })


    except mysql.connector.Error as error:
        print("Error while retrieving data from MySQL: {}".format(error))
        return "No userData found!"

    finally:
        mydb.close()