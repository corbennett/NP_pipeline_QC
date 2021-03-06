# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:30:43 2020

@author: svc_ccg
"""

from psycopg2 import connect, extras


def query_lims(query_string):
    
    con = connect(
        dbname='lims2',
        user='limsreader',
        host='limsdb2',
        password='limsro',
        port=5432,
        )
    con.set_session(
        readonly=True, 
        autocommit=True,
        )
    cursor = con.cursor(
        cursor_factory=extras.RealDictCursor,
                )

    
    cursor.execute(query_string)
    result = cursor.fetchall()
    
    return result