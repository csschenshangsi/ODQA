#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime

from common import logger
from common import tf
from common.builder import get_estimator
from common.builder import input_fn_builder
from common.config import get_config_from_args
from features import get_record_parser
from tasks import get_task_model_class


def train(config):
    start = datetime.now()
    tf.set_random_seed(config.random_seed or 1)
    num_train_steps = int(config.dataset_size / config.train_batch_size  # 79693/32*3=7471
                          * config.epochs)
    # can set to config
    config.num_train_steps = num_train_steps  # 7471
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)  # 7471*0.15=1120
    config.num_warmup_steps = num_warmup_steps  # 1120

    # print config
    logger.info('config: \n{}'.format(
        '\n'.join(['{}: {}'.format(i[0], i[1])
                   for i in sorted(config.items())])))
    model_class = get_task_model_class(config.model, config.task)  # ABCMeta, config.model=bert, task=squad-v1.1
    estimator, model = get_estimator(config, model_class)  # estimator=TPUEstimator, model= tasks.qa_bert_BertQa
    record_parser = get_record_parser(config.model, config.task)  # QaDataBuilder.record_parser
    train_input_fn = input_fn_builder(record_parser, config)
    logger.info('begin training for {} steps....'.format(num_train_steps))

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)  ####
    end = datetime.now()

    logger.info('training ended!')
    logger.info('all done, took {} s!'.format(end - start))


def main(args):
    config = get_config_from_args(args, mode='train')
    train(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-m', '--model', type=str, default='sbert',
                        choices=('bert', 'ebert', 'sbert'),
                        help='choose model to load default configuration')
    parser.add_argument('-t', '--task', type=str, default='squad_v1.1',
                        choices=('squad_v1.1', 'mnli', 'qqp', 'boolq', 'race'),
                        help='choose model to load default configuration')
    main(parser.parse_args())
