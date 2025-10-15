import json
import csv

def convert_json_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    headers = [
        'operator_name',
        'device',
        'call_count',
        'avg_time_ms',
        'total_time_ms',
        'input_shape_1',
        'input_shape_2',
        'output_shapes'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        # for every operator
        for operator in data['operator_statistics']:
            operator_name = operator['operator_name']
            device = operator['device']

            # for every call
            for call in operator['calls']:
                call_count = call['call_count']
                avg_time_ms = call['avg_time_ms']
                total_time_ms = call['total_time_ms']

                # process input_shapes，divide into two columns
                input_shapes = call['input_shapes']
                input_shape_1 = str(input_shapes[0]) if len(input_shapes) > 0 else ""
                input_shape_2 = str(input_shapes[1]) if len(input_shapes) > 1 else ""

                # process output_shapes，keep them in one column
                output_shapes = '; '.join([str(shape) for shape in call['output_shapes']])

                # write CSV row
                writer.writerow([
                    operator_name,
                    device,
                    call_count,
                    avg_time_ms,
                    total_time_ms,
                    input_shape_1,
                    input_shape_2,
                    output_shapes
                ])

    print(f"Convert success! CSV file is saved: {output_file}")

if __name__ == "__main__":
    input_json = "operator_stats.json"
    output_csv = "operator_stats.csv"
    convert_json_to_csv(input_json, output_csv)