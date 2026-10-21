"""Offline comparison of the two user-executed ID3 probes; no serial imports."""
import hashlib
import json
from pathlib import Path

root = Path('/home/yu/so101_ws')
names = [
    'data/real_control_runs/elbow_probe_20261020T225204_rltw_nqz.json',
    'data/real_control_runs/elbow_probe_20261020T225702__du71an6.json',
]
cases = []
for name in names:
    raw = (root/name).read_bytes()
    run = json.loads(raw)
    g0, p0 = run['original_targets']['3'], run['loaded_baseline']
    goal = g0-4
    samples = [r for r in run['reads'] if r['phase']=='step_-4'
               and r['servo_id']==3 and r['address']==56
               and r['command_target']==goal]
    baseline = [r for r in run['reads'] if r['phase']=='loaded_baseline'
                and r['servo_id']==3 and r['address']==56]
    control = [r for r in run['reads'] if r['phase']=='step_-4'
               and r['servo_id']==3 and r['address']==41]
    decoded = []
    for row in control:
        b = bytes.fromhex(row['raw_hex'])
        values = {'acceleration': b[0], 'goal':int.from_bytes(b[1:3],'little'),
                  'speed':int.from_bytes(b[5:7],'little'),
                  'limit':int.from_bytes(b[7:9],'little')}
        if values['goal']==goal:
            decoded.append(values)
    config = bytes.fromhex(next(r['raw_hex'] for r in run['reads']
                                if r['servo_id']==3 and r['address']==0))
    cases.append({
        'run':name, 'source_sha256':hashlib.sha256(raw).hexdigest(),
        'g0':g0, 'p0':p0, 'baseline_bias':p0-g0, 'goal':goal,
        'baseline_output_values':sorted({r['output_raw_signed'] for r in baseline}),
        'after_command_sample_count':len(samples),
        'sample_span_s':[samples[0]['elapsed_s'],samples[-1]['elapsed_s']],
        'position_values':sorted({r['position'] for r in samples}),
        'first_position':samples[0]['position'], 'last_position':samples[-1]['position'],
        'last_relative_displacement':samples[-1]['position']-p0,
        'output_values':sorted({r['output_raw_signed'] for r in samples}),
        'last_output':samples[-1]['output_raw_signed'],
        'speed_values':sorted({r['speed_raw_signed'] for r in samples}),
        'expected_feedback_interval':[p0-6,p0-2],
        'samples_in_arrival_band':sum(p0-6<=r['position']<=p0-2 for r in samples),
        'control_readbacks':len(decoded),
        'control_readbacks_expected':all(c=={'acceleration':1,'goal':goal,'speed':50,'limit':100} for c in decoded),
        'id4_positions':sorted({r['position'] for r in run['reads'] if r['phase']=='step_-4' and r['servo_id']==4 and r['address']==56}),
        'status_values':sorted({r['status'] for r in samples}),
        'device_config':{'P':config[21],'D':config[22],'I':config[23],
                         'minimum_starting_output':config[24],
                         'dead_zones':[config[26],config[27]]},
        'release':run['release'],
    })
report = {
    'schema_version':1,
    'kind':'offline_comparison_of_id3_loaded_and_partly_supported_probes',
    'assistant_serial_access':False,
    'cases':cases,
    'observations':[
        'The first test has zero sampled displacement; the second has only -1 count relative to its fixed loaded baseline.',
        'Both first steps correctly timed out without issuing the second motion target, and all controlled IDs reported torque/output off.',
        'Both control readbacks match their intended parameters; the second output settles at 40 below the fixed cap of 100.',
    ],
    'hypothesis_not_diagnosis':'Small-step response may be limited by breakaway/friction/load characteristics; these records do not isolate the physical cause.',
    'next_diagnostic':{
        'single_step_total_counts':-8,
        'baseline_bias_max_abs_counts':2,
        'fresh_prewrite_position_minus_g0_max_abs_counts':2,
        'output_cap':100,
        'timeout_s':3,
        'final_relative_feedback_interval_counts':[-10,-6],
        'wrist_drift_limit_counts':3,
        'automatic_return':False,
        'pid_or_eeprom_changes':False,
        'scope':'One attended partly-supported response test; full joint mapping and grasp remain unverified.',
    },
    'register_reference':'https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation',
    'output_units_note':'The documented load feedback is motor drive PWM duty cycle, not calibrated mechanical torque.',
}
destination = Path('/tmp/so101_elbow_comparison_stage/comparison.json')
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps([{'g0':c['g0'],'p0':c['p0'],'samples':c['after_command_sample_count'],
                   'relative_movement':c['last_relative_displacement'],
                   'steady_output':c['last_output']} for c in cases]))
