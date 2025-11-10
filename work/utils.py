import os

import time
import json
import joblib
import re
from collections import Counter
from pathlib import Path
import math

import numpy as np
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt

import random

def seconds_to_time(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = round(seconds % 60, 2)

    result = {
        'input_seconds': seconds,
        'hours': hours,
        'minutes': minutes,
        'seconds': remaining_seconds,
        'text': f'{hours} jam {minutes} menit {remaining_seconds} detik'
    }
    return result

def save_json(data, json_file):  
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def flatten_dict(nested_dict, parent_key='', sep='.'):  
    items = {}  
    for key, value in nested_dict.items():  
        new_key = f"{parent_key}{sep}{key}" if parent_key else key  
        if isinstance(value, dict):  
            items.update(flatten_dict(value, new_key, sep=sep))  
        else:  
            items[new_key] = value  
    return items

def load_json(file_path):
    try:  
        with open(file_path, 'r') as file:  
            data = json.load(file)  
        return data  
    except FileNotFoundError:  
        print(f"Error: The file '{file_path}' was not found.")  
    except json.JSONDecodeError:  
        print(f"Error: The file '{file_path}' is not a valid JSON file.")  
    except Exception as e:  
        print(f"An error occurred: {e}")

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def save_object(obj, filename, compress=9):  
    with open(filename, 'wb') as file:  
        joblib.dump(obj, file, compress=compress)

def load_object(filename):  
    with open(filename, 'rb') as file:  
        obj = joblib.load(file)    
    return obj

def index_ranges(data, n_items=[]):
    n = len(data)
    k = len(n_items)

    if k < 2:
        raise ValueError(f"Minimal 2 n_items")

    middle_n_items = n_items[1 : k - 1]
    divisor = len(middle_n_items) + 1
    distance = int((n - np.sum(n_items)) / divisor)

    ranges = []
    ranges.append((0, n_items[0]))
    current_index = n_items[0]
    for ni in middle_n_items:
        current_index += distance
        ranges.append((current_index, current_index + ni))
        current_index += ni
    ranges.append((n - n_items[k - 1], n))

    return ranges

def printhtml(html):
    display(HTML(html))

def html_br():
    printhtml('<br>')

def display_table(data, table_style='width: 100%', column_widths=[], text_aligns=[], hidden_columns=[], n_items=[], max_texts=[], save_excel=None):
    """
    Menampilkan tabel HTML di Jupyter Lab.

    Parameters:
    - data (list of dict): Data yang akan ditampilkan dalam tabel.
    - table_style (str): Gaya CSS untuk tabel (default: 'width: 100%').
    - column_widths (list): Daftar lebar kolom dalam persen (misalnya ['10%', '20%', '70%']).
                            Jika kosong, tidak ada lebar kolom yang diterapkan.
    - text_aligns (list): Daftar perataan teks untuk setiap kolom (misalnya ['left', 'center', 'right']).
                          Jika kosong, semua kolom akan menggunakan perataan default 'left'.
    """
    if not data:
        print("Data kosong. Tidak ada yang ditampilkan.")
        return

    # Ambil header dari keys dictionary pertama
    headers = list(data[0].keys())
    headers = [h for h in headers if h not in hidden_columns]

    if len(n_items) < 2:
        new_data = data.copy()
    else:
        new_data = []
        for start_idx, end_idx in index_ranges(data, n_items):
            new_data += data[start_idx:end_idx]
            new_data.append({h: '...' for h in headers})
        new_data = new_data[:len(new_data) - 1]

    if save_excel is not None:
        pd.DataFrame(new_data).to_excel(save_excel, index=False)

    # Pastikan text_aligns memiliki nilai default jika kosong
    if not text_aligns:
        text_aligns = ['left'] * len(headers)

    # Mulai membuat tabel HTML
    html = f'<table style="{table_style}; border-collapse: collapse;">\n'

    # Tambahkan baris header
    html += '  <tr>\n'
    for i, header in enumerate(headers):
        # Tambahkan lebar kolom jika column_widths diberikan
        width_style = f"width: {column_widths[i]};" if column_widths and i < len(column_widths) else ""
        # Tambahkan perataan teks
        align_style = f"text-align: {text_aligns[i]};"
        html += f'    <th style="border: 1px solid black; padding: 8px; {width_style} {align_style}">{header}</th>\n'
    html += '  </tr>\n'

    # Tambahkan baris data
    for row in new_data:
        html += '  <tr>\n'
        for i, key in enumerate(headers):
            content = row[key]
            if key in max_texts:
                pass
            
            # Tambahkan lebar kolom jika column_widths diberikan
            width_style = f"width: {column_widths[i]};" if column_widths and i < len(column_widths) else ""
            # Tambahkan perataan teks
            align_style = f"text-align: {text_aligns[i]};"
            html += f'    <td style="border: 1px solid black; padding: 8px; {width_style} {align_style}">{row[key]}</td>\n'
        html += '  </tr>\n'

    # Akhiri tabel HTML
    html += '</table>'

    # Tampilkan tabel menggunakan IPython.display
    printhtml(html)

    return new_data