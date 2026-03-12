from pydantic import BaseModel
from typing import List, Optional

class DocumentMeta(BaseModel):
    doc_number: Optional[str] = None
    doc_date: Optional[str] = None

class Participants(BaseModel):
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    payer: Optional[str] = None
    carrier: Optional[str] = None

class VehicleInfo(BaseModel):
    car_brand: Optional[str] = None
    license_plate: Optional[str] = None
    driver_name: Optional[str] = None

class GoodsItem(BaseModel):
    item_id: Optional[str] = None
    name: Optional[str] = None
    unit: Optional[str] = None
    quantity: Optional[str] = None
    price_per_unit: Optional[str] = None
    total_price: Optional[str] = None

class Totals(BaseModel):
    total_quantity: Optional[str] = None
    total_amount: Optional[str] = None

class TTNDocument(BaseModel):
    document_meta: DocumentMeta = DocumentMeta()
    participants: Participants = Participants()
    vehicle_info: VehicleInfo = VehicleInfo()
    goods_table: List[GoodsItem] = []
    totals: Totals = Totals()