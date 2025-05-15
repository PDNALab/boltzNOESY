import yaml
import pandas as pd
from pathlib import Path

# Custom representer to force [A] style for id
def represent_list_flowstyle(dumper, data):
    if all(isinstance(x, str) for x in data) and len(data) == 1:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

yaml.add_representer(list, represent_list_flowstyle)

def parse_a3m(a3m_path):
    with open(a3m_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    chain_id = None
    sequence = []
    for line in lines:
        if line.startswith('>'):
            chain_id = line[1:].split()[0]
        else:
            sequence.append(line)
    return chain_id, ''.join(sequence)

def process_noesy(noesy_path, sequence):
    df = pd.read_csv(noesy_path, sep='\t', header=None,
                     names=['res_from', 'res_to', 'peakID', 'distance', 'atom_from', 'atom_to'])
    constraints = []
    for _, row in df.iterrows():
        res_from = int(row['res_from'])
        res_to = int(row['res_to'])
        if not (1 <= res_from <= len(sequence) and 1 <= res_to <= len(sequence)):
            continue
        constraints.append({
            'noesy': {
                'residue_from': res_from,
                'residue_to': res_to,
                'distance': float(row['distance']),
                'atom_from': 'CA',
                'atom_to': 'CA'
            }
        })
    return constraints

def create_config(a3m_path, noesy_path, output_path):
    chain_id, seq = parse_a3m(a3m_path)
    constraints = process_noesy(noesy_path, seq)
    yaml_data = {
        'version': 1,
        'sequences': [
            {
                'protein': {
                    'id': [chain_id],
                    'sequence': seq,
                    'msa': str(Path(a3m_path).absolute())
                }
            }
        ],
        'constraints': constraints
    }

    # Dump YAML with block style
    yaml_str = yaml.dump(
        yaml_data,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        width=1000
    )

    # Fix atom_from and atom_to quoting and indentation
    lines = yaml_str.splitlines()
    new_lines = []
    for line in lines:
        # Remove any blank lines (except between top-level keys)
        if line.strip() == '':
            continue
        # Properly quote atom_from and atom_to as "CA"
        if 'atom_from:' in line or 'atom_to:' in line:
            parts = line.split(':')
            line = f"{parts[0]}: \"{parts[1].strip().strip('\"') }\""
        new_lines.append(line)

    # Remove blank lines between sequences: and its first item
    # And between constraints: and its first item
    formatted_yaml = '\n'.join(new_lines)
    with open(output_path, 'w') as f:
        f.write(formatted_yaml+'\n')

# Example usage
if __name__ == "__main__":
    create_config(
        a3m_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/test.a3m",
        noesy_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/T0953s2_AmbiR.txt",
        output_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/noesy.yaml"
    )
