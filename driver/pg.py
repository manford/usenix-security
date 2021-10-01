import logging
import psycopg2
import psycopg2.extras
import time

from urllib.parse import urlparse


class PG:
    def __init__(self, url=None):
        self.url = url
        self.config = None
        if url:
            self.parseUrl(self.url)
        try:
            self.init()
        except Exception as e:
            logging.error('Init PG error!')
            logging.error(e)

    def init(self):
        # self.pool = psycopg2.pool.ThreadedConnectionPool(
        #     minconn=5,
        #     maxconn=20,
        #     host=self.hostname,
        #     dbname=self.database,
        #     user=self.username,
        #     password=self.password,
        #     port=self.port,
        # )
        # logging.info('PG connected!')
        return self

    def query(self, sql, param = None):
        conn = psycopg2.connect(
            host=self.hostname,
            dbname=self.database,
            user=self.username,
            password=self.password,
            port=self.port,
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            start = time.time()
            cursor.execute(sql,param)
            end = time.time()
            print("query time : " , end - start)
        except Exception as e:
            logging.error('PG exec error! SQL: ' + sql)
            logging.error(e)
        rows = cursor.fetchall()
        output = []
        for row in rows:
            json_row = {}
            for key in row:
                json_row[key] = row[key]
            output.append(json_row)
        cursor.close()
        conn.close()
        return output

    def update(self, sql, param = None):
        conn = psycopg2.connect(
            host=self.hostname,
            dbname=self.database,
            user=self.username,
            password=self.password,
            port=self.port,
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cursor.execute(sql, param)
        except Exception as e:
            logging.error('PG exec error! SQL: ' + sql)
            logging.error(e)
        conn.commit()
        res = cursor.statusmessage
        cursor.close()
        conn.close()
        return res

    def close(self):
        pass

    def parseUrl(self, url):
        params = urlparse(url)
        self.username = params.username
        self.password = params.password
        self.database = params.path[1:]
        self.hostname = params.hostname
        self.port = params.port
        if self.port == None:
            self.port = 5432
