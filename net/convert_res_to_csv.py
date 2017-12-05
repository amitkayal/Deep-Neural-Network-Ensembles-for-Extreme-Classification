import pandas as pd
from Utils import *

org_test_csv_path = ""
org_test_csv = pd.read_csv(org_test_csv_path)
internal_images_ids = list(org_test_csv['image_id'])

res_path = ""


for internal_image_id in internal_images_ids:
    product_id = imageid_to_productid(internal_image_id)