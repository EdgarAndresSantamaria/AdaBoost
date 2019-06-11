#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sys

class Exporter:

    def __init__(self,file):
        self.FILE = file

    def save_data(self,data):
        try:
            with open(self.FILE, "wb") as f:
                pickle.dump(data, f)
        except:
            print("Se ha producido un error al exportar.")
            sys.out(0)
