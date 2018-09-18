from elasticsearch_dsl import (DocType, Text)

from . import settings


class News(DocType):
    source = Text()
    url = Text()
    content = Text()

    class Index:
        name = 'news'

    def save(self, ** kwargs):
        return super().save(** kwargs)
