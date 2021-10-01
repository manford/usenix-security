import logging
import pika


class RMQ:
    def __init__(self, url=None, queue=None, handler=None):
        if not url:
            logging.error('RMQ: url required!')
            return
        if not queue:
            logging.error('RMQ: queue name required!')
            return
        if not handler:
            logging.error('RMQ: message handler required!')
            return
        self.url = url
        self.queue = queue
        self.handler = handler
        self.channel = None

    def init(self):
        logging.info('Get RMQ addr ' + self.url)
        logging.info('Connect to RMQ server...')
        self.parameters = pika.URLParameters(self.url)
        connection = pika.SelectConnection(self.parameters, self.on_connected)
        try:
            logging.info('Start Consume message...')
            connection.ioloop.start()
        except KeyboardInterrupt:
            connection.close()
            # connection.ioloop.start()

    def on_connected(self, connection):
        logging.info('Connected to RMQ server.')
        connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, new_channel):
        logging.info('Connected to Queue ' + self.queue)
        self.channel = new_channel
        self.channel.queue_declare(queue=self.queue,
                                   durable=True,
                                   exclusive=False,
                                   auto_delete=False,
                                   callback=self.on_queue_declared)

    def on_queue_declared(self, frame):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.queue, self.handler)
