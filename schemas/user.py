from pydantic import BaseModel
from pydantic import BaseModel, Field, EmailStr
class User(BaseModel):
    id: int
    name: str
    email: str
    password: str


class UserLoginSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)
    fullname: str | None = Field(None)
    class Config:
        schema_extra = {
            "example": {
                "email": "joe@xyz.com",
                "password": "any"
            }
        }


class UserSchema(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.id = None

    fullname: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)
    role: str = Field(...)


    class Config:
        schema_extra = {
            "example": {
                "fullname": "Joe Doe",
                "email": "joe@xyz.com",
                "password": "any",
                "role": "user"
            }
        }


