"""Create a separately versioned host guard; do not alter v1 evidence."""
from pathlib import Path
p=Path(__file__).with_name('d02a_ram_guard.py')
source=p.read_text().replace("OUT/'resource_guard_handoff.json'", "OUT/'resource_guard_handoff_v2.json'").replace("OUT/'resource_guard'", "OUT/'resource_guard_v2'")
needle="                        index=worker_index(command)"
assert needle in source
source=source.replace(needle,"""                        # Windows may expose a CREATE_SUSPENDED child before
                        # subprocess has resumed its initial thread. Let the
                        # parent finish spawning before holding its next launch.
                        cpu=candidate.cpu_times()
                        if cpu.user+cpu.system==0:
                            time.sleep(.02);continue
                        index=worker_index(command)""")
p.with_name('d02a_ram_guard_v2.py').open('x').write(source)
