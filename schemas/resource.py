from pydantic import BaseModel, Field, EmailStr


class Resources(BaseModel):
    res_id: int
    title: str
    desc: str
    link: str
    image: str
    type: str
    field: str


class Ratings(BaseModel):
    res_id: int
    user_id: int
    rate: int

