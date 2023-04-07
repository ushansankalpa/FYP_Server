import mysql.connector
import hashlib
import hmac

from auth.auth_handler import signJWT


def registerUser(Userdata, learningstyle):
    mydb = mysql.connector.connect(
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
        query = "SELECT * FROM users WHERE email=%s AND password=%s"
        mycursor.execute(query, (userData.email, hashed_password))
        result = mycursor.fetchone()
        # if result:
        #
        #     return True
        # else:
        #
        #     return False

        if result:
            # Return user ID along with login success status
            return (True, result[0])
        else:
            return (False, None)

    except mysql.connector.Error as error:
        print("Error while querying data from MySQL: {}".format(error))
        return False

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
        sql = "SELECT id, fullname, email, role FROM users WHERE email = %s"
        mycursor.execute(sql, (user_id,))
        result = mycursor.fetchone()

        if result:
            user_data = {
                "id": result[0],
                "fullname": result[1],
                "email": result[2],
                "role": result[3]
            }

            return signJWT(user_data)
        else:
            return "No userData found!"

    except mysql.connector.Error as error:
        print("Error while retrieving data from MySQL: {}".format(error))
        return "No userData found!"

    finally:
        mydb.close()


def updateUserMeasurement(data):
    try:
        sql = "UPDATE usermeasurements SET age = %s, neck = %s, knee = %s, ankle = %s, biceps = %s, forearm = %s, wrist = %s, weight = %s, height = %s, abdomen = %s, chest = %s, hip = %s, thigh = %s WHERE userId = %s"
        val = (
        data.age, data.neck, data.knee, data.ankle, data.biceps, data.forearm, data.wrist, data.weight, data.height,
        data.abdomen, data.chest, data.hip, data.thigh, data.userId)
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
