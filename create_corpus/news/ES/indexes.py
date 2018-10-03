from datetime import datetime
from elasticsearch_dsl import (DocType, Text, Keyword, Integer)

from . import settings


class News(DocType):
    source = Keyword()
    url = Text()
    content = Text()
    length = Integer()
    cons = Integer()
    preudo = Integer()
    factual = Keyword()
    notes = Text()
    update = Text()

    class Index:
        name = 'news'

    def save(self, ** kwargs):
        self.created_at = datetime.now()
        return super().save(** kwargs)
