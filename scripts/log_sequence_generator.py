import json
import random
import datetime
import uuid
from typing import List, Dict, Any, Optional

class LogSequenceGenerator:
    def __init__(self):
        # API-like service names
        self.services = [
            "api_user", "api_auth", "api_payment", "api_order", "api_inventory",
            "api_notification", "api_analytics", "api_search", "api_file", "api_config"
        ]
        
        self.log_levels = ["INFO", "WARN", "ERROR", "DEBUG", "FATAL"]
        
        self.normal_operations = [
            "login", "register", "create_order", "process_payment", "query_inventory",
            "upload_file", "update_config", "fetch_data", "update_cache", "health_check",
            "api_call", "db_connect", "send_message", "log_entry", "record_metrics"
        ]
        
        self.error_operations = [
            "db_connect_fail", "network_timeout", "out_of_memory", "disk_full",
            "auth_fail", "permission_denied", "service_unavailable", "data_format_error",
            "concurrent_conflict", "deadlock", "resource_exhausted", "config_error", "dependency_error"
        ]
        
        self.error_messages = [
            "Connection refused: database server is down",
            "Timeout after 30 seconds waiting for response",
            "OutOfMemoryError: Java heap space",
            "No space left on device",
            "Authentication failed: invalid credentials",
            "Permission denied: insufficient privileges",
            "Service unavailable: 503 error",
            "Invalid data format: expected JSON",
            "Concurrent modification detected",
            "Deadlock detected in transaction",
            "Resource exhausted: too many connections",
            "Configuration error: missing required parameter",
            "Dependency service error: upstream service unavailable"
        ]
        
        self.normal_messages = [
            "Request processed successfully",
            "User authenticated successfully",
            "Order created with ID: {order_id}",
            "Payment processed successfully",
            "Inventory updated successfully",
            "File uploaded successfully",
            "Configuration updated",
            "Data retrieved successfully",
            "Cache updated successfully",
            "Health check passed",
            "API call completed",
            "Database connection established",
            "Message sent successfully",
            "Log entry created",
            "Performance metrics recorded"
        ]
        
        self.service_dependencies = {
            "api_order": ["api_user", "api_inventory", "api_payment"],
            "api_payment": ["api_auth", "api_user"],
            "api_notification": ["api_user", "api_order"],
            "api_analytics": ["api_order", "api_user"],
            "api_search": ["api_file", "api_user"],
            "api_file": ["api_auth"],
            "api_inventory": ["api_config"]
        }
    
    def generate_timestamp(self, base_time=None, offset_range=300):
        if base_time is None:
            base_time = datetime.datetime.now()
        offset = random.randint(-offset_range, offset_range)
        return base_time + datetime.timedelta(seconds=offset)
    
    def generate_log_entry_string(self, service_name: str, is_error: bool = False, 
                                 request_id: Optional[str] = None,
                                 user_id: Optional[str] = None) -> str:
        """Generate a single log entry as a formatted string"""
        timestamp = self.generate_timestamp()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Format: 2025-06-25 09:15:30.123
        
        if is_error:
            log_level = random.choice(["ERROR", "FATAL", "WARN"])
            operation = random.choice(self.error_operations)
            message = random.choice(self.error_messages)
        else:
            log_level = random.choice(["INFO", "DEBUG"])
            operation = random.choice(self.normal_operations)
            message = random.choice(self.normal_messages)
            if "{order_id}" in message:
                message = message.format(order_id=str(uuid.uuid4())[:8])
        
        # Generate request_id and user_id if not provided
        req_id = request_id or str(uuid.uuid4())[:8]
        usr_id = user_id or f"user_{random.randint(1000, 9999)}"
        
        # Create formatted log string
        log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] {message}"
        
        # Add request_id and user_id to some messages
        if random.random() < 0.3:  # 30% chance to include request_id
            log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] Request {req_id} {message.lower()}"
        elif random.random() < 0.2:  # 20% chance to include user_id
            log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] User {usr_id} {message.lower()}"
        
        return log_string
    
    def generate_normal_sequence_strings(self, sequence_length: int) -> List[str]:
        """Generate normal sequence as list of log strings"""
        request_id = str(uuid.uuid4())[:8]
        user_id = f"user_{random.randint(1000, 9999)}"
        primary_service = random.choice(self.services)
        sequence = []
        
        # Start request
        sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
        
        # Authentication if needed
        if primary_service in ["api_order", "api_payment", "api_file"]:
            sequence.append(self.generate_log_entry_string("api_auth", False, request_id, user_id))
        
        # Dependency service calls
        if primary_service in self.service_dependencies:
            for dep_service in self.service_dependencies[primary_service]:
                if random.random() < 0.7:
                    sequence.append(self.generate_log_entry_string(dep_service, False, request_id, user_id))
        
        # Main business logic
        for _ in range(max(0, sequence_length - len(sequence))):
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
        
        # Complete request
        sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
        
        return sequence
    
    def generate_anomaly_sequence_strings(self, sequence_length: int) -> List[str]:
        """Generate anomaly sequence as list of log strings"""
        request_id = str(uuid.uuid4())[:8]
        user_id = f"user_{random.randint(1000, 9999)}"
        
        anomaly_type = random.choice([
            "service_cascade_failure",
            "dependency_failure", 
            "resource_exhaustion",
            "timeout_cascade",
            "authentication_failure",
            "concurrent_conflict"
        ])
        
        sequence = []
        
        if anomaly_type == "service_cascade_failure":
            primary_service = random.choice(self.services)
            failed_service = random.choice(self.services)
            
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            if failed_service in self.service_dependencies.get(primary_service, []):
                sequence.append(self.generate_log_entry_string(failed_service, True, request_id, user_id))
                sequence.append(self.generate_log_entry_string(primary_service, True, request_id, user_id))
            
            for service in self.services:
                if service != primary_service and service != failed_service:
                    if random.random() < 0.4:
                        sequence.append(self.generate_log_entry_string(service, random.choice([True, False]), request_id, user_id))
        
        elif anomaly_type == "dependency_failure":
            primary_service = random.choice(["api_order", "api_payment", "api_notification"])
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            for dep_service in self.service_dependencies.get(primary_service, []):
                sequence.append(self.generate_log_entry_string(dep_service, True, request_id, user_id))
            
            sequence.append(self.generate_log_entry_string(primary_service, True, request_id, user_id))
        
        elif anomaly_type == "resource_exhaustion":
            services = random.sample(self.services, 3)
            for i, service in enumerate(services):
                if i == 0:
                    sequence.append(self.generate_log_entry_string(service, True, request_id, user_id))
                else:
                    sequence.append(self.generate_log_entry_string(service, random.choice([True, False]), request_id, user_id))
        
        elif anomaly_type == "timeout_cascade":
            primary_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            for service in random.sample(self.services, random.randint(2, 4)):
                sequence.append(self.generate_log_entry_string(service, True, request_id, user_id))
        
        elif anomaly_type == "authentication_failure":
            sequence.append(self.generate_log_entry_string("api_auth", True, request_id, user_id))
            
            for service in ["api_order", "api_payment", "api_file"]:
                sequence.append(self.generate_log_entry_string(service, True, request_id, user_id))
        
        elif anomaly_type == "concurrent_conflict":
            primary_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            sequence.append(self.generate_log_entry_string(primary_service, True, request_id, user_id))
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            sequence.append(self.generate_log_entry_string(primary_service, True, request_id, user_id))
        
        # Fill to required length
        while len(sequence) < sequence_length:
            service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(service, random.choice([True, False]), request_id, user_id))
        
        return sequence
    
    def generate_dataset_strings(self, num_sequences: int = 8000, anomaly_ratio: float = 0.3) -> List[Dict[str, Any]]:
        """Generate dataset with log entries as strings"""
        dataset = []
        num_anomalies = int(num_sequences * anomaly_ratio)
        num_normal = num_sequences - num_anomalies
        
        print(f"Generating {num_normal} normal sequences...")
        for i in range(num_normal):
            if i % 500 == 0:
                print(f"Progress: {i}/{num_normal}")
            sequence_length = random.randint(6, 15)
            log_strings = self.generate_normal_sequence_strings(sequence_length)
            dataset.append({
                "sequence_id": str(uuid.uuid4()),
                "label": 0,
                "length": len(log_strings),
                "log_entries": log_strings
            })
        
        print(f"Generating {num_anomalies} anomaly sequences...")
        for i in range(num_anomalies):
            if i % 500 == 0:
                print(f"Progress: {i}/{num_anomalies}")
            sequence_length = random.randint(6, 15)
            log_strings = self.generate_anomaly_sequence_strings(sequence_length)
            dataset.append({
                "sequence_id": str(uuid.uuid4()),
                "label": 1,
                "length": len(log_strings),
                "log_entries": log_strings
            })
        
        random.shuffle(dataset)
        return dataset
    
    def save_string_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset with log entries as strings"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"String format dataset saved to {filename}")
        print(f"Total sequences: {len(dataset)}")
        print(f"Normal sequences: {sum(1 for item in dataset if item['label'] == 0)}")
        print(f"Anomaly sequences: {sum(1 for item in dataset if item['label'] == 1)}")

def main():
    generator = LogSequenceGenerator()
    print("Generating log sequence dataset...")
    
    # Generate dataset with string format - 8000 samples
    dataset_strings = generator.generate_dataset_strings(num_sequences=8000, anomaly_ratio=0.3)
    generator.save_string_dataset(dataset_strings, "data/log_sequences_strings.json")
    
    print("\nSample normal sequence (string format):")
    normal_sequence = generator.generate_normal_sequence_strings(8)
    for i, log in enumerate(normal_sequence):
        print(f"  {i+1}. {log}")
    
    print("\nSample anomaly sequence (string format):")
    anomaly_sequence = generator.generate_anomaly_sequence_strings(8)
    for i, log in enumerate(anomaly_sequence):
        print(f"  {i+1}. {log}")

if __name__ == "__main__":
    main() 