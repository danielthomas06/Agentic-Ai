from pydantic import BaseModel


class TwoNumbers(BaseModel):
    a: float
    b: float
