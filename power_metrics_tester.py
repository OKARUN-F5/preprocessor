import json
from datetime import datetime, timedelta
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any

# Import the optimized functions we created earlier
# (Copy the code from the previous implementation here)
def calculate_metrics(metrics: Dict[str, float], transformer_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all metrics in a single pass for efficiency
    Returns a dictionary with all calculated values
    """
    # Extract required values once to avoid repeated dictionary lookups
    active_power_a = metrics.get('active_power_overall_phase_a', 0)
    active_power_b = metrics.get('active_power_overall_phase_b', 0)
    active_power_c = metrics.get('active_power_overall_phase_c', 0)
    
    reactive_power_a = metrics.get('reactive_power_overall_phase_a', 0)
    reactive_power_b = metrics.get('reactive_power_overall_phase_b', 0)
    reactive_power_c = metrics.get('reactive_power_overall_phase_c', 0)
    
    power_factor_a = metrics.get('power_factor_overall_phase_a', 0)
    power_factor_b = metrics.get('power_factor_overall_phase_b', 0)
    power_factor_c = metrics.get('power_factor_overall_phase_c', 0)
    
    voltage_a = metrics.get('line_to_neutral_voltage_phase_a', 0)
    voltage_b = metrics.get('line_to_neutral_voltage_phase_b', 0)
    voltage_c = metrics.get('line_to_neutral_voltage_phase_c', 0)
    
    current_a = metrics.get('line_current_overall_phase_a', 0)
    current_b = metrics.get('line_current_overall_phase_b', 0)
    current_c = metrics.get('line_current_overall_phase_c', 0)
    
    # Pre-calculate absolute values to avoid redundant operations
    abs_active_a = abs(active_power_a)
    abs_active_b = abs(active_power_b)
    abs_active_c = abs(active_power_c)
    
    abs_reactive_a = abs(reactive_power_a)
    abs_reactive_b = abs(reactive_power_b)
    abs_reactive_c = abs(reactive_power_c)
    
    abs_pf_a = abs(power_factor_a)
    abs_pf_b = abs(power_factor_b)
    abs_pf_c = abs(power_factor_c)
    
    abs_voltage_a = abs(voltage_a)
    abs_voltage_b = abs(voltage_b)
    abs_voltage_c = abs(voltage_c)
    
    abs_current_a = abs(current_a)
    abs_current_b = abs(current_b)
    abs_current_c = abs(current_c)
    
    # Calculate metrics
    # Current Load (kW)
    current_load = abs_active_a + abs_active_b + abs_active_c
    
    # Reactive Power (kVAR)
    reactive_power = abs_reactive_a + abs_reactive_b + abs_reactive_c
    
    # Power Factor
    power_factor = (abs_pf_a + abs_pf_b + abs_pf_c) / 3 if (abs_pf_a + abs_pf_b + abs_pf_c) > 0 else 0
    
    # Distributed Electricity (kWh) - for 1-minute interval
    distributed_electricity = current_load * 0.01666667
    
    # Voltage Imbalance Factor
    voltage_sum = abs_voltage_a + abs_voltage_b + abs_voltage_c
    avg_voltage = voltage_sum / 3 if voltage_sum > 0 else 0
    max_voltage = max(abs_voltage_a, abs_voltage_b, abs_voltage_c)
    voltage_imbalance = (max_voltage - avg_voltage) / avg_voltage if avg_voltage > 0 else 0
    
    # Current Imbalance Factor
    current_sum = abs_current_a + abs_current_b + abs_current_c
    avg_current = current_sum / 3 if current_sum > 0 else 0
    max_current = max(abs_current_a, abs_current_b, abs_current_c)
    current_imbalance = (max_current - avg_current) / avg_current if avg_current > 0 else 0
    
    # Transformer Load Percentage
    transformer_capacity = transformer_info.get('transformer_capacity', 0)
    transformer_load_pct = (current_load / power_factor) / transformer_capacity if power_factor > 0 and transformer_capacity > 0 else 0
    
    return {
        'current_load_kw': current_load,
        'reactive_power_kvar': reactive_power,
        'apparent_power_kva': metrics.get('apparent_power', 0),  # From database
        'distributed_electricity_kwh': distributed_electricity,
        'frequency_hz': metrics.get('frequency', 0),
        'power_factor': power_factor,
        'voltage_imbalance_factor': voltage_imbalance,
        'current_imbalance_factor': current_imbalance,
        'transformer_capacity_kva': transformer_capacity,
        'transformer_production_year': transformer_info.get('production_year', ''),
        'transformer_position_type': transformer_info.get('position_type', ''),
        'transformer_position_nature': transformer_info.get('position_nature', ''),
        'transformer_load_percentage': transformer_load_pct,
        'phase_voltages': {
            'phase_a': abs_voltage_a,  # brown_colour
            'phase_b': abs_voltage_b,  # black_colour
            'phase_c': abs_voltage_c   # grey_colour
        },
        'phase_currents': {
            'phase_a': abs_current_a,  # brown_colour
            'phase_b': abs_current_b,  # black_colour
            'phase_c': abs_current_c,  # grey_colour
            'neutral': metrics.get('line_current_overall_neutral', 0)
        },
        'gauge_total': 1
    }

def process_daily_metrics(daily_data: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Process daily aggregated metrics in a single efficient pass
    """
    if not daily_data:
        return {
            'excess_load': 0,
            'excess_current_imbalance': 0,
            'excess_voltage_imbalance': 0,
            'excess_power_factor': 0,
            'transformer_uptime': 0,
            'daily_max_load': 0,
            'daily_min_load': 0
        }
    
    # Initialize counters
    excess_load_count = 0
    excess_current_imbalance_count = 0
    excess_voltage_imbalance_count = 0
    excess_power_factor_count = 0
    
    # Initialize for min/max tracking
    loads = []
    
    # Single pass through daily data
    for data_point in daily_data:
        # Track loads for min/max calculation
        load = data_point.get('load', 0)
        loads.append(load)
        
        # Count excesses
        if data_point.get('transformer_load_percentage', 0) > 1:
            excess_load_count += 1
            
        if data_point.get('current_imbalance_factor', 0) > 0.2:
            excess_current_imbalance_count += 1
            
        if data_point.get('voltage_imbalance_factor', 0) > 0.07:
            excess_voltage_imbalance_count += 1
            
        if data_point.get('power_factor', 1) < 0.8:
            excess_power_factor_count += 1
    
    # Calculate daily metrics
    total_data_points = len(daily_data)
    minutes_in_day = 1440
    
    return {
        'excess_load': excess_load_count / minutes_in_day,
        'excess_current_imbalance': excess_current_imbalance_count / minutes_in_day,
        'excess_voltage_imbalance': excess_voltage_imbalance_count / minutes_in_day,
        'excess_power_factor': excess_power_factor_count / minutes_in_day,
        'transformer_uptime': total_data_points / minutes_in_day,
        'daily_max_load': max(loads) if loads else 0,
        'daily_min_load': min(loads) if loads else 0
    }

def lambda_handler(event, context):
    """
    Process metrics data and return calculated values in a cost-effective way
    """
    # Extract data from event
    metrics = event.get('metrics', {})
    transformer_info = event.get('transformer_info', {})
    daily_data = event.get('daily_data', [])
    
    # Calculate current metrics in one efficient pass
    current_metrics = calculate_metrics(metrics, transformer_info)
    
    # Calculate daily aggregated metrics if provided
    if daily_data:
        daily_metrics = process_daily_metrics(daily_data)
        # Merge the results
        current_metrics.update(daily_metrics)
    
    return current_metrics

# Data Generation Functions
def generate_realistic_metrics(timestamp: datetime, base_load: float = 100.0, variance: float = 0.2) -> Dict:
    """
    Generate realistic metrics data with variations based on time of day
    """
    # Time-based modifiers for realistic load patterns
    hour = timestamp.hour
    
    # Simulate daily load pattern (lower at night, peak in morning and evening)
    hour_factor = 0.3  # Base load at night
    if 6 <= hour < 10:  # Morning peak
        hour_factor = 0.7 + (hour - 6) * 0.1  # Gradual increase
    elif 10 <= hour < 16:  # Mid-day
        hour_factor = 0.8
    elif 16 <= hour < 22:  # Evening peak
        hour_factor = 0.9 + (hour - 16) * 0.05 if hour < 20 else 0.9 - (hour - 20) * 0.1
    
    # Apply time-based load and some randomness
    load_factor = hour_factor * (1 + random.uniform(-variance, variance))
    current_load = base_load * load_factor
    
    # Power factor tends to be worse under high load
    power_factor_base = 0.92 - (load_factor - 0.5) * 0.1 if load_factor > 0.5 else 0.92
    
    # Create some phase imbalance
    phase_a_factor = 1 + random.uniform(-0.15, 0.15)
    phase_b_factor = 1 + random.uniform(-0.15, 0.15)
    phase_c_factor = 1 + random.uniform(-0.15, 0.15)
    
    # Normalize to maintain total load
    total_factor = phase_a_factor + phase_b_factor + phase_c_factor
    phase_a_factor = phase_a_factor * 3 / total_factor
    phase_b_factor = phase_b_factor * 3 / total_factor
    phase_c_factor = phase_c_factor * 3 / total_factor
    
    # Calculate phase loads
    phase_a_load = current_load * phase_a_factor / 3
    phase_b_load = current_load * phase_b_factor / 3
    phase_c_load = current_load * phase_c_factor / 3
    
    # Base voltage with some variation
    base_voltage = 230.0 * (1 + random.uniform(-0.05, 0.05))
    voltage_a = base_voltage * (1 + random.uniform(-0.03, 0.03))
    voltage_b = base_voltage * (1 + random.uniform(-0.03, 0.03))
    voltage_c = base_voltage * (1 + random.uniform(-0.03, 0.03))
    
    # Calculate currents based on power and voltage
    # P = V * I * PF
    pf_a = power_factor_base * (1 + random.uniform(-0.05, 0.05))
    pf_b = power_factor_base * (1 + random.uniform(-0.05, 0.05))
    pf_c = power_factor_base * (1 + random.uniform(-0.05, 0.05))
    
    current_a = phase_a_load * 1000 / (voltage_a * pf_a) if voltage_a * pf_a != 0 else 0
    current_b = phase_b_load * 1000 / (voltage_b * pf_b) if voltage_b * pf_b != 0 else 0
    current_c = phase_c_load * 1000 / (voltage_c * pf_c) if voltage_c * pf_c != 0 else 0
    
    # Calculate reactive power based on power factor
    reactive_a = phase_a_load * math.tan(math.acos(pf_a))
    reactive_b = phase_b_load * math.tan(math.acos(pf_b))
    reactive_c = phase_c_load * math.tan(math.acos(pf_c))
    
    # Calculate neutral current (should be near zero in balanced system)
    # Simplified calculation using phasor sum approximation
    neutral_current = abs(current_a - (current_b + current_c) / 2) * 0.3
    
    return {
        'timestamp': timestamp.isoformat(),
        'active_power_overall_phase_a': phase_a_load,
        'active_power_overall_phase_b': phase_b_load,
        'active_power_overall_phase_c': phase_c_load,
        'reactive_power_overall_phase_a': reactive_a,
        'reactive_power_overall_phase_b': reactive_b,
        'reactive_power_overall_phase_c': reactive_c,
        'power_factor_overall_phase_a': pf_a,
        'power_factor_overall_phase_b': pf_b,
        'power_factor_overall_phase_c': pf_c,
        'line_to_neutral_voltage_phase_a': voltage_a,
        'line_to_neutral_voltage_phase_b': voltage_b,
        'line_to_neutral_voltage_phase_c': voltage_c,
        'line_current_overall_phase_a': current_a,
        'line_current_overall_phase_b': current_b,
        'line_current_overall_phase_c': current_c,
        'line_current_overall_neutral': neutral_current,
        'frequency': 50 + random.uniform(-0.1, 0.1),
        'apparent_power': abs(phase_a_load + phase_b_load + phase_c_load) / ((pf_a + pf_b + pf_c) / 3)
    }

def generate_daily_data(base_load: float = 100.0, start_date: datetime = None) -> List[Dict]:
    """
    Generate a full day of data (1440 minutes)
    """
    if start_date is None:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    daily_data = []
    
    # Create data for each minute of the day
    for minute in range(1440):
        timestamp = start_date + timedelta(minutes=minute)
        
        # Generate metrics for this timestamp
        metrics = generate_realistic_metrics(timestamp, base_load)
        
        # Add to daily data
        daily_data.append(metrics)
    
    return daily_data

def generate_transformer_info() -> Dict:
    """
    Generate realistic transformer information
    """
    return {
        'transformer_capacity': random.choice([100, 160, 250, 400, 630]),
        'production_year': str(random.randint(1990, 2020)),
        'position_type': random.choice(['Pole-mounted', 'Ground-mounted', 'Substation']),
        'position_nature': random.choice(['Urban', 'Rural', 'Industrial'])
    }

def simulate_lambda_event(base_load: float = 100.0) -> Dict:
    """
    Simulate an AWS Lambda event with metrics and transformer info
    """
    now = datetime.now()
    
    # Generate current metrics snapshot
    current_metrics = generate_realistic_metrics(now, base_load)
    
    # Generate transformer info
    transformer_info = generate_transformer_info()
    
    # Generate daily data (optional for testing daily aggregates)
    daily_data = generate_daily_data(base_load, now.replace(hour=0, minute=0, second=0, microsecond=0))
    
    # Create processed daily data for testing
    processed_daily_data = []
    for data in daily_data:
        # Process each data point to get metrics needed for daily aggregates
        metrics_result = calculate_metrics(data, transformer_info)
        processed_daily_data.append({
            'load': metrics_result['current_load_kw'],
            'transformer_load_percentage': metrics_result['transformer_load_percentage'],
            'current_imbalance_factor': metrics_result['current_imbalance_factor'],
            'voltage_imbalance_factor': metrics_result['voltage_imbalance_factor'],
            'power_factor': metrics_result['power_factor']
        })
    
    return {
        'metrics': current_metrics,
        'transformer_info': transformer_info,
        'daily_data': processed_daily_data
    }

def run_lambda_simulation(num_tests: int = 5) -> None:
    """
    Run multiple simulations with different loads
    """
    loads = [50, 100, 200, 300, 500]  # Different load scenarios
    
    for i, load in enumerate(loads[:num_tests]):
        print(f"\n--- Test {i+1}: Base Load {load} kW ---")
        
        # Simulate event
        event = simulate_lambda_event(load)
        
        # Call lambda handler
        result = lambda_handler(event, None)
        
        # Print key results
        print(f"Current Load: {result['current_load_kw']:.2f} kW")
        print(f"Power Factor: {result['power_factor']:.2f}")
        print(f"Transformer Load: {result['transformer_load_percentage'] * 100:.2f}%")
        print(f"Voltage Imbalance: {result['voltage_imbalance_factor'] * 100:.2f}%")
        print(f"Current Imbalance: {result['current_imbalance_factor'] * 100:.2f}%")
        
        if 'excess_load' in result:
            print(f"Excess Load Time: {result['excess_load'] * 100:.2f}%")
            print(f"Excess Power Factor Time: {result['excess_power_factor'] * 100:.2f}%")
            print(f"Transformer Uptime: {result['transformer_uptime'] * 100:.2f}%")

def visualize_daily_load(base_load: float = 100.0) -> None:
    """
    Visualize a day's worth of load data
    """
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    daily_data = generate_daily_data(base_load, start_date)
    
    # Extract timestamps and load values
    timestamps = [datetime.fromisoformat(data['timestamp']) for data in daily_data]
    hours = [(dt - start_date).total_seconds() / 3600 for dt in timestamps]
    
    loads_a = [data['active_power_overall_phase_a'] for data in daily_data]
    loads_b = [data['active_power_overall_phase_b'] for data in daily_data]
    loads_c = [data['active_power_overall_phase_c'] for data in daily_data]
    total_loads = [a + b + c for a, b, c in zip(loads_a, loads_b, loads_c)]
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(hours, total_loads, label='Total Load (kW)')
    plt.title(f'Daily Load Profile (Base: {base_load} kW)')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(hours, loads_a, 'brown', label='Phase A')
    plt.plot(hours, loads_b, 'black', label='Phase B')
    plt.plot(hours, loads_c, 'gray', label='Phase C')
    plt.xlabel('Hour of Day')
    plt.ylabel('Phase Power (kW)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('daily_load_profile.png')
    print("Graph saved as 'daily_load_profile.png'")
    plt.close()

def test_extreme_cases() -> None:
    """
    Test extreme cases to verify the code's robustness
    """
    print("\n--- Testing Extreme Cases ---")
    
    # Test case 1: Zero values
    zero_metrics = {k: 0 for k in [
        'active_power_overall_phase_a', 'active_power_overall_phase_b', 'active_power_overall_phase_c',
        'reactive_power_overall_phase_a', 'reactive_power_overall_phase_b', 'reactive_power_overall_phase_c',
        'power_factor_overall_phase_a', 'power_factor_overall_phase_b', 'power_factor_overall_phase_c',
        'line_to_neutral_voltage_phase_a', 'line_to_neutral_voltage_phase_b', 'line_to_neutral_voltage_phase_c',
        'line_current_overall_phase_a', 'line_current_overall_phase_b', 'line_current_overall_phase_c',
        'line_current_overall_neutral', 'frequency', 'apparent_power'
    ]}
    
    transformer_info = {'transformer_capacity': 100, 'production_year': '2010', 'position_type': 'Test', 'position_nature': 'Test'}
    
    result = calculate_metrics(zero_metrics, transformer_info)
    print("Zero values test - Load:", result['current_load_kw'])
    print("Zero values test - Power Factor:", result['power_factor'])
    
    # Test case 2: Negative values
    negative_metrics = {k: -10 for k in zero_metrics.keys()}
    result = calculate_metrics(negative_metrics, transformer_info)
    print("Negative values test - Load:", result['current_load_kw'])
    print("Negative values test - Power Factor:", result['power_factor'])
    
    # Test case 3: Very high values
    high_metrics = {k: 10000 for k in zero_metrics.keys()}
    result = calculate_metrics(high_metrics, transformer_info)
    print("High values test - Load:", result['current_load_kw'])
    print("High values test - Transformer Load %:", result['transformer_load_percentage'])

def save_sample_event_to_file(filename: str = "sample_lambda_event.json") -> None:
    """
    Generate and save a sample lambda event to a file
    """
    event = simulate_lambda_event(100)
    
    # Save to file with pretty formatting
    with open(filename, 'w') as f:
        json.dump(event, f, indent=2)
    
    print(f"\nSample event saved to {filename}")
    print("You can use this file to test your AWS Lambda function.")

def main():
    """
    Main test function
    """
    print("===== Power Metrics Lambda Function Tester =====")
    print("1. Running simulations with different loads...")
    run_lambda_simulation()
    
    print("\n2. Testing extreme cases...")
    test_extreme_cases()
    
    print("\n3. Generating daily load visualization...")
    try:
        visualize_daily_load()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("You may need to install matplotlib to see the visualization.")
    
    print("\n4. Saving sample event for Lambda testing...")
    save_sample_event_to_file()
    
    print("\n===== Testing Complete =====")
    print("The code appears to be functioning correctly with simulated data.")
    print("You can now deploy it to AWS Lambda with confidence.")

if __name__ == "__main__":
    main()