#!/usr/bin/env python
"""
A simple wrapper around snakemake to run it with a slurm cluster and default configs
"""

import click
import subprocess


@click.command()
@click.argument('snakefile', required=False, type=click.Path(exists=True))
@click.argument('rule', required=False, type=str)
@click.option('--snakefile', '-s', 'snakefile_opt', help='Snakefile to run', required=False, type=click.Path(exists=True))
@click.option('--cores', '-j', help='Number of cores to use', default=10, type=int)
@click.option('--dry-run', '-n', help='Dry run', is_flag=True)
@click.option('--profile', '-p', help='Profile to use', default='slurm_configs/default', type=click.Path(exists=True))
@click.option('--cluster-config', '-c', help='Cluster config to use', default='slurm_configs/cluster.yaml', type=click.Path(exists=True))
@click.option('--dag', '-d', help='Output the DAG', type=click.Path(exists=False))
def run(snakefile, rule, snakefile_opt, cores, dry_run, profile, cluster_config, dag):
    snakefile = snakefile_opt if snakefile_opt else snakefile
    if snakefile is None:
        raise click.UsageError('No Snakefile provided')
    command = [
        'snakemake',
        '-s', snakefile
    ]
    if not rule is None:
        command.append(rule)
    command += [
        '-j', str(cores),
        '--profile', str(profile),
        '--cluster-config', str(cluster_config)
    ]
    if dry_run:
        command.append('-n')
    if dag:
        print('piping to dag')
        command.append('--dag')
        # print(command)
        p1 = subprocess.Popen(command, stdout=subprocess.PIPE)
        command = ["dot", "-Tpng"]
        # print(command)
        with open(dag, 'wb') as f:
            p2 = subprocess.Popen(command, stdin=p1.stdout, stdout=f)
            p1.stdout.close()
        p2.wait()

    else:
        print('running command ' + ' '.join(command))
        result = subprocess.run(command, text=True)

if __name__ == '__main__':
    run()

