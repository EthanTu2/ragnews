import sqlite3
db = sqlite3.connect('ragnews.db')
#b.row_factory = sqlite3.Row
cursor = db.cursor()

sql = '''
SELECT title, text, bm25(articles) AS rank
FROM articles
WHERE articles MATCH 'trump Harris debate'
ORDER BY rank
LIMIT 10;
'''
cursor.execute(sql)
rows = cursor.fetchall()
for row in rows:
    print(f"row={row[0]}\Content: {row[1]}")