import yaml
import pandas as pd
from pathlib import Path

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

def process_noesy(noesy_path, sequence, min_seq_sep=1, max_seq_sep=None):
    """
    Only return unique NOESY contacts per residue pair (hydrop, H-H),
    keeping the shortest distance for each pair and filtering by sequence separation.
    """
    df = pd.read_csv(noesy_path, sep='\t', header=None,
                     names=['res_from', 'res_to', 'peakID', 'distance', 'atom_from', 'atom_to'])
    #hydrop = {'I', 'L', 'V','A','M','F','W','P'}
    hydrop = {'I', 'L', 'V','A'}
    filtered = []
    for _, row in df.iterrows():
        res_from = int(row['res_from'])
        res_to = int(row['res_to'])
        seq_sep = abs(res_from - res_to)
        if not (1 <= res_from <= len(sequence) and 1 <= res_to <= len(sequence)):
            continue
        aa_from = sequence[res_from - 1]
        aa_to = sequence[res_to - 1]
        atom_from = str(row['atom_from']).strip().upper()
        atom_to = str(row['atom_to']).strip().upper()
        # Sequence separation filtering
        if (aa_from in hydrop) and (aa_to in hydrop) and (atom_from == "H") and (atom_to == "H"):
            if seq_sep >= min_seq_sep and (max_seq_sep is None or seq_sep <= max_seq_sep):
                filtered.append({
                    'res_from': res_from,
                    'res_to': res_to,
                    'distance': float(row['distance']),
                    'seq_sep': seq_sep,
                    'atom_from': "CA",
                    'atom_to': "CA"
                })

    if not filtered:
        return []

    # Only keep one contact per unique residue pair (regardless of direction), with shortest distance
    unified_contacts = {}
    for c in filtered:
        pair = tuple(sorted((c['res_from'], c['res_to'])))
        current_distance = c['distance']
        if pair not in unified_contacts or current_distance < unified_contacts[pair]['distance']:
            unified_contacts[pair] = {
                'res_from': pair[0],
                'res_to': pair[1],
                'distance': current_distance,
                'atom_from': "CA",
                'atom_to': "CA"
            }

    # Convert to output format
    return [{
        'noesy': {
            'residue_from': c['res_from'],
            'residue_to': c['res_to'],
            'distance': c['distance'],
            'atom_from': c['atom_from'],
            'atom_to': c['atom_to']
        }
    } for c in unified_contacts.values()]

def create_config(a3m_path, noesy_path, output_path, min_seq_sep=1, max_seq_sep=None):
    chain_id, seq = parse_a3m(a3m_path)
    constraints = process_noesy(noesy_path, seq, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep)
    yaml_data = {
        'version': 1,
        'sequences': [{
            'protein': {
                'id': [chain_id],
                'sequence': seq,
                'msa': str(Path(a3m_path).absolute())
            }
        }],
        'constraints': constraints
    }
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False, indent=2)

if __name__ == "__main__":
    # Example: Only allow contacts with sequence separation >= 5 and <= 30
    create_config(
        a3m_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/test.a3m",
        noesy_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/T0953s2_AmbiR.txt",
        output_path="/orange/alberto.perezant/imesh.ranaweera/boltz-noesy_install/noesy_filtered.yaml",
        min_seq_sep=2,   # set your minimum sequence separation here
        max_seq_sep=100   # or set to None if you don't want an upper limit
    )
