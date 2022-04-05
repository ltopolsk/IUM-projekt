from pydantic import BaseModel
from typing import Optional


class Session(BaseModel):
    session_id: int
    timestamp: str
    user_id: int
    product_id: int
    event_type: str
    offered_discount: int
    purchase_id: Optional[int] = None


class User(BaseModel):
    user_id: int
    name: str
    city: str
    street: str


class Delivery(BaseModel):
    purchase_id: int
    purchase_timestamp: str
    delivery_timestamp: str
    delivery_company: int


class Product(BaseModel):
    product_id: int
    product_name: str
    category_path: str
    price: int

