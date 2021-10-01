import numpy as np
import os
import sys
sys.path.insert(0, './driver')
import driver.pg as pg
import json
import psycopg2

os.getenv('NEBULA_AI_PG_READ_URL')

url = 'postgresql://ai_r:IglhreXEE1BiEB6q@pgm-2ze587qcqvvo29xylo.pg.rds.aliyuncs.com:3433/platform-ai'

pg_dw = pg.PG(url)

result_limit = 10



serialNumber = "SN19030223"
equipmentPartId = "2b67729a-13d8-4f61-bc0e-fea21a77be21"


sql_dw = """
        SELECT "sensor"->>'acceleration' as "acceleration" ,"sensor"->'plc'->>'flow' as "flow",
        "label"->>'operating' as "operating","label"->>'anomaly' as "anomaly",
        "label"->>'condition' as "condition","node"->>'serialNumber' as "serialNumber",
        "equipment"->>'parameter' as  "parameter" , "equipment"->>'equipmentName' as  "equipmentName"
        FROM "Point"
        WHERE "node"->>'serialNumber'=%s AND "equipment"->>'equipmentPartId'=%s
        ORDER BY "timestamp" DESC limit %s
        """

sql_0 = """
SELECT * from "Point" limit 100
"""


dataT = pg_dw.query(sql_0, (serialNumber, equipmentPartId, str(result_limit), ))

print("Returned data length is", len(dataT))
print(dataT)


