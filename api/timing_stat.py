import time
from typing import Dict, Optional
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

class TimingStats:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stats: Dict[str, float] = {}
        self.timers: Dict[str, float] = {}
    
    def start_timer(self, stage_name: str) -> None:
        self.timers[stage_name] = time.time()
        print(f"{Fore.CYAN}[{self.request_id}] ⏱️  Started: {stage_name}{Style.RESET_ALL}")
    
    def end_timer(self, stage_name: str) -> float:
        if stage_name not in self.timers:
            print(f"{Fore.YELLOW}[{self.request_id}] ⚠️  No start timer for {stage_name}{Style.RESET_ALL}")
            return 0.0
        
        elapsed = time.time() - self.timers[stage_name]
        self.stats[stage_name] = elapsed
        del self.timers[stage_name]
        if elapsed < 2:
            color = Fore.GREEN
            status = "✓ FAST"
        elif elapsed < 5:
            color = Fore.YELLOW
            status = "⚡ MODERATE"
        else:
            color = Fore.RED
            status = "⏳ SLOW"
        
        print(f"{color}[{self.request_id}] {status}: {stage_name} took {elapsed:.2f}s{Style.RESET_ALL}")
        return elapsed
    
    def print_summary(self) -> None:
        if not self.stats:
            return
        
        print(f"\n{Back.BLUE}{Fore.WHITE}{'═' * 70}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}📊 TIMING STATS SUMMARY [{self.request_id}]{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * 70}{Style.RESET_ALL}\n")
        
        for stage, duration in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(duration * 10)
            bar = "█" * min(bar_length, 40)
            
            print(f"{Fore.CYAN}├─ {stage:<40} {Fore.YELLOW}{duration:>8.2f}s {bar}{Style.RESET_ALL}")
        
        total_time = sum(self.stats.values())
        print(f"\n{Fore.GREEN}└─ {'TOTAL TIME':<40} {Fore.WHITE}{Back.GREEN}{total_time:>8.2f}s{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * 70}{Style.RESET_ALL}\n")
    
    def add_stat(self, stage_name: str, duration: float) -> None:
        self.stats[stage_name] = duration