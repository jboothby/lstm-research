import json
import sys

def main(file_name):
    labels = [0, 1]
    with open(f'inference-summary-binary-{file_name}.json', 'r') as f:
        x = json.load(f)
    data = x['Testing']['data']

    with open(f'latex_table-binary-{file_name}.txt', 'w') as out:
        for k, v in data.items():
            print('\\hline', file=out)
            print(f'{k} & ', end='', file=out)
            total = 0
            for val in data[k].values():
                total += val
            print(f'{total}', end='', file=out)
            for label in labels:
                if label in data[k].keys():
                    print(f' & {data[k][label]}', end='', file=out)
                else:
                    print(f' & 0', end='', file=out)
            print('\\\\', file=out)

        print('\\hline', file=out)

    out.close()


if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    main(sys.argv[1])






