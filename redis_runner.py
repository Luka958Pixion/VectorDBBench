import subprocess


def get_command(
    db: str,
    host: str,
    password: str,
    port: int,
    ssl: bool,
    case_type: str, 
    m: int, 
    ef_construction: int, 
    ef_search: int, 
    k: int, 
    dry_run: bool,
    num_concurrency: list[int],
    concurrency_duration: int,
    drop_old: bool,
    load: bool,
    search_serial: bool,
    search_concurrent: bool
) -> list:
    items = {
        '/home/lukap/VectorDBBench/env/bin/vectordbbench': db,
        '--host': host,
        '--password': password,
        '--port': str(port),
        '--case-type': case_type,
        '--m': str(m),
        '--ef-construction': str(ef_construction),
        '--ef-runtime': str(ef_search),
        '--k': str(k),
        '--num-concurrency': ','.join(map(str, num_concurrency)),
        '--concurrency-duration': str(concurrency_duration)
    }
    args = [x for item in items.items() for x in item]
    
    if dry_run:
        args.append('--dry-run')
        
    args.append('--drop-old' if drop_old else '--skip-drop-old')
    args.append('--load' if load else '--skip-load')
    args.append('--search-serial' if search_serial else '--skip-search-serial')
    args.append('--search-concurrent' if search_concurrent else '--skip-search-concurrent')
    args.append('--ssl' if ssl else '--no-ssl')
    
    return args

def run_cli_app(command):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, shell=False)
        print(f'STDOUT:\n {result.stdout}\nSTDERR:\n{result.stderr}\n')
            
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")


# main
EF_SEARCH_START = 16

common = {
    'case_type': 'Performance1536D50K',
    'k': 16,
    'dry_run': False,
    'num_concurrency': [1],
    'concurrency_duration': 30,
    'search_serial': True,
    'search_concurrent': True
}

connection = {
    'db': 'redis',
    'host': 'localhost',
    'password': 'password',
    'port': 6379,
    'ssl': False
}

commands = [
    get_command(
        **common,
        **connection,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        drop_old=ef_search == EF_SEARCH_START,
        load=ef_search == EF_SEARCH_START
    )
    for ef_construction in [256] #[64, 128, 256]
    for m in range(4, 64 + 1, 4)    # TODO back to 64
    for ef_search in range(16, 128 + 1, 4)
]

for command in commands:
    run_cli_app(command)
        