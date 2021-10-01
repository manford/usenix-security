# Pi2star AI Driver

---

## pg

数据库模块使用Demo

```python
from drvier.pg import PG

pg_dw = None

url = 'postgresql://dw_readonly:F8NTzIvGwEhNgqOk@192.168.1.243:5433/dw'

def main():
    global pg_dw
    pg_dw = PG(url) # 仅算法开发测试时需初始化，生产环境下直接调用全局 pg_dw 对象即可
    
    result_limit = 3

    sql = f"""
            SELECT
	            *
            FROM
                "EquipmentHealth"
	        LIMIT {result_limit}
            """
    dataset = pg_dw.query(sql)
    print(len(dataset))
    print(dataset)

if __name__ == '__main__':
    main()
```



