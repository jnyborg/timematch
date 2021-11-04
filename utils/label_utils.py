import os
import csv
import yaml

def get_classes(*countries, method=set.union, combine_spring_and_winter=False):
    class_sets = []
    for country in countries:
        code_to_class = get_code_to_class(country, combine_spring_and_winter)
        class_sets.append(set(code_to_class.values()))

    classes = sorted(list(method(*class_sets)))
    return classes

def read_yaml_class_mapping(country):
    return yaml.load(open(os.path.join('class_mapping', f'{country}_class_mapping.yml')), Loader=yaml.FullLoader)

def get_code_to_class(country, combine_spring_and_winter=False):
    class_mapping = read_yaml_class_mapping(country)

    code_to_class = {}
    for cls in class_mapping.keys():
        codes = class_mapping[cls]
        if codes is None:
            continue
        if 'spring' in codes and 'winter' in codes:
            if combine_spring_and_winter:
                combined = {**(codes['spring'] if codes['spring'] is not None else {}), **(codes['winter'] if codes['winter'] is not None else {})}
                code_to_class.update({code: cls for code in combined})
            else:
                if codes['spring'] is not None:
                    code_to_class.update({code: f'spring_{cls}' for code in codes['spring'].keys()})
                if codes['winter'] is not None:
                    code_to_class.update({code: f'winter_{cls}' for code in codes['winter'].keys()})
        else:
            code_to_class.update({code: cls for code in codes})
    return code_to_class

def get_shapefile_columns(country):
    cols = _shapefile_columns[country]
    return cols['id'], cols['crop_code']

def get_codification_table(country):
    codification_table = os.path.join('class_mapping', f'{country}_codification_table.csv')
    with open(codification_table, newline='') as f:
        delimiter = ';' if country in ['denmark', 'austria'] else ','
        crop_codes = csv.reader(f, delimiter=delimiter)
        crop_codes = {x[0]: x[1] for x in crop_codes}  # crop_code: (name, group)
    return crop_codes

_shapefile_columns = {
    'denmark': {
        'id': 'marknr',
        'crop_code': 'afgkode',
    },
    'france': {
        'id': 'id_parcel',
        'crop_code': 'code_cultu',
    },
    'austria': {
        'id': 'geo_id',
        'crop_code': 'snar_code',
    }
}
