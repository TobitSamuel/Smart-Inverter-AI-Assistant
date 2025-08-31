import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
warnings.filterwarnings('ignore')

# -----------------------------
# DC/DC Converter Class
# -----------------------------
class DCDCConverter:
    """AI-Enhanced DC/DC Converter for Solar-EV Interface"""

    def __init__(self):
        self.input_voltage_range = (30, 50)  # V (solar panel range)
        self.output_voltage_range = (300, 800)  # V (DC link range)
        self.power_rating = 12000  # W
        self.efficiency = 0.96
        self.switching_frequency = 50000  # Hz
        self.mppt_enabled = True

    def boost_convert(self, v_in, i_in, duty_cycle):
        """Boost converter operation"""
        if duty_cycle >= 1.0:
            return {'output_voltage': 0, 'output_current': 0, 'output_power': 0, 'efficiency': 0}
            
        v_out = v_in / (1 - duty_cycle)
        i_out = i_in * (1 - duty_cycle) * self.efficiency
        power_out = v_out * i_out

        return {
            'output_voltage': v_out,
            'output_current': i_out,
            'output_power': power_out,
            'efficiency': self.efficiency
        }

    def buck_convert(self, v_in, i_in, duty_cycle):
        """Buck converter operation"""
        if duty_cycle <= 0.01:
            return {'output_voltage': 0, 'output_current': 0, 'output_power': 0, 'efficiency': 0}
            
        v_out = v_in * duty_cycle
        i_out = i_in / duty_cycle * self.efficiency
        power_out = v_out * i_out

        return {
            'output_voltage': v_out,
            'output_current': i_out,
            'output_power': power_out,
            'efficiency': self.efficiency
        }

# -----------------------------
# EV Motor Drive Class
# -----------------------------
class EVMotorDrive:
    """Electric Vehicle Motor Drive System"""

    def __init__(self):
        self.motor_type = "PMSM"
        self.rated_power = 10000
        self.rated_voltage = 400
        self.rated_current = 25
        self.rated_speed = 3000
        self.rated_torque = 32
        self.efficiency = 0.94

        self.pole_pairs = 4
        self.stator_resistance = 0.05
        self.d_inductance = 0.8e-3
        self.q_inductance = 1.2e-3
        self.flux_linkage = 0.15

        self.speed_ref = 0
        self.torque_ref = 0
        self.current_speed = 0
        self.current_torque = 0

    def calculate_motor_currents(self, torque_ref, speed_ref):
        id_ref = 0
        iq_ref = torque_ref / (1.5 * self.pole_pairs * self.flux_linkage)
        i_magnitude = np.sqrt(id_ref ** 2 + iq_ref ** 2)

        return {
            'id_ref': id_ref,
            'iq_ref': iq_ref,
            'current_magnitude': i_magnitude,
            'power': torque_ref * speed_ref * 2 * np.pi / 60
        }

    def motor_dynamics(self, torque_electric, load_torque, dt=0.001):
        J = 0.01
        net_torque = torque_electric - load_torque
        speed_change = net_torque / J * dt
        self.current_speed += speed_change * 60 / (2 * np.pi)

        mech_power = self.current_speed * torque_electric * 2 * np.pi / 60
        elec_power = mech_power / self.efficiency

        return {
            'speed_rpm': self.current_speed,
            'torque_nm': torque_electric,
            'mechanical_power': mech_power,
            'electrical_power': elec_power
        }

# -----------------------------
# Smart EV Inverter Class
# -----------------------------
class SmartEVInverter:
    """AI-Enhanced Three-Phase Smart Inverter for EV Motor Drive"""

    def __init__(self):
        self.voltage_rating = 400
        self.power_rating = 12000
        self.frequency_range = (0, 200)
        self.switching_frequency = 10000

        self.mppt_optimizer = None
        self.fault_classifier = None
        self.motor_control_optimizer = None
        self.efficiency_optimizer = None

        self.efficiency = 0.95
        self.thd_limit = 0.05

        self.vector_control_enabled = True
        self.field_weakening_enabled = True

        self._initialize_ai_models()

    def _initialize_ai_models(self):
        print("Initializing AI models for EV motor drive...")

        self.mppt_optimizer = RandomForestClassifier(n_estimators=100, random_state=42)
        self.fault_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.motor_control_optimizer = RandomForestClassifier(n_estimators=75, random_state=42)
        self.efficiency_optimizer = RandomForestClassifier(n_estimators=100, random_state=42)

        self._train_mppt_model()
        self._train_fault_detection_model()
        self._train_motor_control_model()
        self._train_efficiency_model()

        print("AI models for EV drive initialized successfully!")

    def _train_mppt_model(self):
        np.random.seed(42)
        n_samples = 1200

        irradiance = np.random.uniform(200, 1000, n_samples)
        temperature = np.random.uniform(15, 45, n_samples)
        voltage = np.random.uniform(30, 50, n_samples)
        current = np.random.uniform(5, 30, n_samples)
        motor_load = np.random.uniform(0, 100, n_samples)

        X = np.column_stack([irradiance, temperature, voltage, current, motor_load])

        duty_cycle = (irradiance/1000 * 0.6 + (50-temperature)/50 * 0.2 +
                      voltage/50 * 0.1 + motor_load/100 * 0.1) * 100
        duty_cycle = np.clip(duty_cycle, 10, 90)
        
        # Convert to classification problem by binning duty cycles
        duty_cycle_bins = np.digitize(duty_cycle, bins=np.linspace(10, 90, 20))
        
        self.mppt_optimizer.fit(X, duty_cycle_bins)

    def _train_fault_detection_model(self):
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data for fault detection
        voltage = np.random.uniform(300, 800, n_samples)
        current = np.random.uniform(0, 30, n_samples)
        temp = np.random.uniform(20, 100, n_samples)
        speed = np.random.uniform(0, 3000, n_samples)
        vibration = np.random.uniform(0, 10, n_samples)
        freq = np.random.uniform(0, 200, n_samples)
        
        X = np.column_stack([voltage, current, temp, speed, vibration, freq])
        
        # Generate synthetic fault labels (0: normal, 1: fault)
        faults = (voltage > 750) | (current > 25) | (temp > 90) | (vibration > 8)
        
        self.fault_classifier.fit(X, faults.astype(int))

    def _train_motor_control_model(self):
        np.random.seed(42)
        n_samples = 1500
        
        # Generate synthetic training data
        speed_ref = np.random.uniform(0, 3000, n_samples)
        torque_ref = np.random.uniform(0, 32, n_samples)
        dc_voltage = np.random.uniform(300, 800, n_samples)
        temperature = np.random.uniform(20, 100, n_samples)
        
        X = np.column_stack([speed_ref, torque_ref, dc_voltage, temperature])
        
        # Generate synthetic control parameters (simplified)
        control_params = np.clip(speed_ref/3000 * 0.8 + torque_ref/32 * 0.2, 0, 1)
        control_bins = np.digitize(control_params, bins=np.linspace(0, 1, 10))
        
        self.motor_control_optimizer.fit(X, control_bins)

    def _train_efficiency_model(self):
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic training data
        power = np.random.uniform(0, 12000, n_samples)
        speed = np.random.uniform(0, 3000, n_samples)
        torque = np.random.uniform(0, 32, n_samples)
        switching_freq = np.random.uniform(5000, 15000, n_samples)
        temperature = np.random.uniform(20, 100, n_samples)
        
        X = np.column_stack([power, speed, torque, switching_freq, temperature])
        
        # Generate synthetic efficiency optimization parameters
        efficiency_params = np.clip(0.85 + power/12000 * 0.1 - temperature/100 * 0.05, 0, 1)
        efficiency_bins = np.digitize(efficiency_params, bins=np.linspace(0.85, 0.95, 10))
        
        self.efficiency_optimizer.fit(X, efficiency_bins)

    def mppt_control_ev(self, irradiance, temperature, voltage, current, motor_load):
        X = np.array([[irradiance, temperature, voltage, current, motor_load]])
        duty_cycle_bin = self.mppt_optimizer.predict(X)[0]
        
        # Convert bin back to actual duty cycle value
        duty_cycle = 10 + (duty_cycle_bin - 1) * (80/20)  # Map bins back to 10-90 range
        return duty_cycle

    def motor_vector_control(self, speed_ref, torque_ref, dc_voltage, temperature):
        X = np.array([[speed_ref, torque_ref, dc_voltage, temperature]])
        control_bin = self.motor_control_optimizer.predict(X)[0]
        
        # Convert bin back to control parameters
        control_param = control_bin / 10.0  # Normalized control parameter
        
        return {
            'modulation_index': control_param * 0.95,
            'current_angle': control_param * 90,
            'switching_pattern': int(control_param * 8)
        }

    def ev_fault_detection(self, dc_voltage, phase_current, motor_temp, motor_speed, vibration, frequency):
        X = np.array([[dc_voltage, phase_current, motor_temp, motor_speed, vibration, frequency]])
        fault_prediction = self.fault_classifier.predict(X)[0]
        
        fault_type = "Normal"
        if fault_prediction == 1:
            if motor_temp > 90:
                fault_type = "Overtemperature"
            elif dc_voltage > 750:
                fault_type = "Overvoltage"
            elif phase_current > 25:
                fault_type = "Overcurrent"
            elif vibration > 8:
                fault_type = "High Vibration"
        
        return {
            'fault_detected': bool(fault_prediction),
            'fault_type': fault_type,
            'severity': 'High' if fault_prediction == 1 else 'Normal'
        }

    def efficiency_optimization(self, power, speed, torque, switching_freq, temperature):
        X = np.array([[power, speed, torque, switching_freq, temperature]])
        efficiency_bin = self.efficiency_optimizer.predict(X)[0]
        
        # Convert bin back to efficiency parameters
        base_efficiency = 0.85 + (efficiency_bin / 10.0) * 0.1
        
        return {
            'predicted_efficiency': base_efficiency,
            'optimal_switching_freq': switching_freq * (1 + (efficiency_bin - 5) / 10),
            'power_factor': 0.95 + (efficiency_bin / 10.0) * 0.05
        }

# -----------------------------
# Example Usage and Testing
# -----------------------------
def test_system():
    """Test the complete EV power system"""
    print("Testing EV Power System Components...")
    
    # Initialize components
    converter = DCDCConverter()
    drive = EVMotorDrive()
    inverter = SmartEVInverter()
    
    print("\n1. Testing DC/DC Converter...")
    try:
        # Test boost conversion
        boost_result = converter.boost_convert(v_in=40, i_in=20, duty_cycle=0.7)
        print(f"Boost Conversion Results: {boost_result}")
        
        # Test buck conversion
        buck_result = converter.buck_convert(v_in=400, i_in=2, duty_cycle=0.5)
        print(f"Buck Conversion Results: {buck_result}")
    except Exception as e:
        print(f"Converter test failed: {str(e)}")
    
    print("\n2. Testing Motor Drive...")
    try:
        # Test motor current calculation
        currents = drive.calculate_motor_currents(torque_ref=20, speed_ref=1500)
        print(f"Motor Currents: {currents}")
        
        # Test motor dynamics
        dynamics = drive.motor_dynamics(torque_electric=25, load_torque=20)
        print(f"Motor Dynamics: {dynamics}")
    except Exception as e:
        print(f"Motor drive test failed: {str(e)}")
    
    print("\n3. Testing Inverter AI Functions...")
    try:
        # Test MPPT control
        mppt_duty = inverter.mppt_control_ev(
            irradiance=800, temperature=25, voltage=40, current=15, motor_load=60
        )
        print(f"MPPT Duty Cycle: {mppt_duty:.2f}%")
        
        # Test fault detection
        fault_status = inverter.ev_fault_detection(
            dc_voltage=400, phase_current=20, motor_temp=70,
            motor_speed=2000, vibration=5, frequency=100
        )
        print(f"Fault Detection: {fault_status}")
        
        # Test efficiency optimization
        efficiency_params = inverter.efficiency_optimization(
            power=8000, speed=2500, torque=25,
            switching_freq=10000, temperature=60
        )
        print(f"Efficiency Parameters: {efficiency_params}")
    except Exception as e:
        print(f"Inverter AI test failed: {str(e)}")
    
    print("\nSystem Test Completed Successfully!")

if __name__ == "__main__":
    test_system()
