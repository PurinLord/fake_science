from elasticsearch_dsl.connections import connections

ELASTICSEARCH = connections.configure(
    default={
        'hosts': 'database:9200',
        'timeout': 60
    },
    sniff_on_start=True
)
