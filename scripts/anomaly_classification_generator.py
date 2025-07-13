import json
import random
import datetime
import uuid
from typing import List, Dict, Any, Optional

class AnomalyClassificationGenerator:
    def __init__(self):
        # API-like service names
        self.services = [
            "api_user", "api_auth", "api_payment", "api_order", "api_inventory",
            "api_notification", "api_analytics", "api_search", "api_file", "api_config"
        ]
        
        self.log_levels = ["INFO", "WARN", "ERROR", "DEBUG", "FATAL"]
        
        # Normal operations
        self.normal_operations = [
            "login", "register", "create_order", "process_payment", "query_inventory",
            "upload_file", "update_config", "fetch_data", "update_cache", "health_check",
            "api_call", "db_connect", "send_message", "log_entry", "record_metrics"
        ]
        
        # Service dependencies
        self.service_dependencies = {
            "api_order": ["api_user", "api_inventory", "api_payment"],
            "api_payment": ["api_auth", "api_user"],
            "api_notification": ["api_user", "api_order"],
            "api_analytics": ["api_order", "api_user"],
            "api_search": ["api_file", "api_user"],
            "api_file": ["api_auth"],
            "api_inventory": ["api_config"]
        }
        
        # Define anomaly categories with specific patterns
        self.anomaly_categories = {
            "service_cascade_failure": {
                "description": "A service failure that cascades to other dependent services",
                "error_messages": [
                    "Connection refused: database server is down",
                    "Service unavailable: 503 error",
                    "Dependency service error: upstream service unavailable",
                    "Circuit breaker opened: too many failures"
                ],
                "pattern": "cascade"
            },
            "authentication_failure": {
                "description": "Authentication and authorization related failures",
                "error_messages": [
                    "Authentication failed: invalid credentials",
                    "Permission denied: insufficient privileges",
                    "Token expired: please re-authenticate",
                    "Access denied: user not authorized"
                ],
                "pattern": "auth"
            },
            "resource_exhaustion": {
                "description": "System resources are exhausted (memory, disk, connections)",
                "error_messages": [
                    "OutOfMemoryError: Java heap space",
                    "No space left on device",
                    "Resource exhausted: too many connections",
                    "Memory allocation failed: insufficient heap space"
                ],
                "pattern": "resource"
            },
            "network_timeout": {
                "description": "Network connectivity and timeout issues",
                "error_messages": [
                    "Timeout after 30 seconds waiting for response",
                    "Connection timeout: remote host not responding",
                    "Network unreachable: connection failed",
                    "Request timeout: service not responding"
                ],
                "pattern": "timeout"
            },
            "data_corruption": {
                "description": "Data integrity and corruption issues",
                "error_messages": [
                    "Invalid data format: expected JSON",
                    "Data corruption detected: checksum mismatch",
                    "Database constraint violation: foreign key error",
                    "Invalid state: data inconsistency detected"
                ],
                "pattern": "data"
            },
            "concurrent_conflict": {
                "description": "Concurrency and race condition issues",
                "error_messages": [
                    "Concurrent modification detected",
                    "Deadlock detected in transaction",
                    "Optimistic lock failed: version conflict",
                    "Transaction rollback: concurrent access conflict"
                ],
                "pattern": "concurrent"
            },
            "configuration_error": {
                "description": "Configuration and setup related errors",
                "error_messages": [
                    "Configuration error: missing required parameter",
                    "Invalid configuration: malformed settings",
                    "Environment variable not found: required config missing",
                    "Configuration validation failed: invalid values"
                ],
                "pattern": "config"
            },
            "database_connection_failure": {
                "description": "Database connectivity and query failures",
                "error_messages": [
                    "Database connection failed: connection refused",
                    "SQL execution failed: query timeout",
                    "Connection pool exhausted: no available connections",
                    "Database server down: connection lost"
                ],
                "pattern": "database"
            }
        }
        
        # Normal messages for comparison
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
    
    def generate_timestamp(self, base_time=None, offset_range=300):
        if base_time is None:
            base_time = datetime.datetime.now()
        offset = random.randint(-offset_range, offset_range)
        return base_time + datetime.timedelta(seconds=offset)
    
    def generate_log_entry_string(self, service_name: str, is_error: bool = False, 
                                 request_id: Optional[str] = None,
                                 user_id: Optional[str] = None,
                                 error_message: Optional[str] = None) -> str:
        """Generate a single log entry as a formatted string"""
        timestamp = self.generate_timestamp()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        if is_error:
            log_level = random.choice(["ERROR", "FATAL", "WARN"])
            message = error_message or "An error occurred"
        else:
            log_level = random.choice(["INFO", "DEBUG"])
            message = random.choice(self.normal_messages)
            if "{order_id}" in message:
                message = message.format(order_id=str(uuid.uuid4())[:8])
        
        # Generate request_id and user_id if not provided
        req_id = request_id or str(uuid.uuid4())[:8]
        usr_id = user_id or f"user_{random.randint(1000, 9999)}"
        
        # Create formatted log string
        log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] {message}"
        
        # Add request_id and user_id to some messages
        if random.random() < 0.3:
            log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] Request {req_id} {message.lower()}"
        elif random.random() < 0.2:
            log_string = f"[{timestamp_str}] [{service_name}] [{log_level}] User {usr_id} {message.lower()}"
        
        return log_string
    
    def generate_normal_sequence(self, sequence_length: int) -> List[str]:
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
    
    def generate_anomaly_sequence(self, anomaly_category: str, sequence_length: int) -> List[str]:
        """Generate anomaly sequence for specific category"""
        request_id = str(uuid.uuid4())[:8]
        user_id = f"user_{random.randint(1000, 9999)}"
        sequence = []
        
        category_info = self.anomaly_categories[anomaly_category]
        error_messages = category_info["error_messages"]
        
        if anomaly_category == "service_cascade_failure":
            # Start with normal operation
            primary_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            # First service fails
            failed_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(
                failed_service, True, request_id, user_id, 
                random.choice(error_messages)
            ))
            
            # Cascade to dependent services
            if failed_service in self.service_dependencies:
                for dep_service in self.service_dependencies[failed_service]:
                    sequence.append(self.generate_log_entry_string(
                        dep_service, True, request_id, user_id,
                        random.choice(error_messages)
                    ))
            
            # Other services also affected
            for service in random.sample(self.services, random.randint(2, 4)):
                if service != failed_service:
                    sequence.append(self.generate_log_entry_string(
                        service, random.choice([True, False]), request_id, user_id,
                        random.choice(error_messages) if random.random() < 0.6 else None
                    ))
        
        elif anomaly_category == "authentication_failure":
            # Auth service fails
            sequence.append(self.generate_log_entry_string(
                "api_auth", True, request_id, user_id,
                random.choice(error_messages)
            ))
            
            # Services that depend on auth also fail
            for service in ["api_order", "api_payment", "api_file", "api_user"]:
                sequence.append(self.generate_log_entry_string(
                    service, True, request_id, user_id,
                    random.choice(error_messages)
                ))
        
        elif anomaly_category == "resource_exhaustion":
            # Multiple services show resource issues
            services = random.sample(self.services, random.randint(3, 5))
            for i, service in enumerate(services):
                if i < 2:  # First two services definitely fail
                    sequence.append(self.generate_log_entry_string(
                        service, True, request_id, user_id,
                        random.choice(error_messages)
                    ))
                else:
                    sequence.append(self.generate_log_entry_string(
                        service, random.choice([True, False]), request_id, user_id,
                        random.choice(error_messages) if random.random() < 0.7 else None
                    ))
        
        elif anomaly_category == "network_timeout":
            # Start normal
            primary_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            # Multiple services timeout
            for service in random.sample(self.services, random.randint(3, 6)):
                sequence.append(self.generate_log_entry_string(
                    service, True, request_id, user_id,
                    random.choice(error_messages)
                ))
        
        elif anomaly_category == "data_corruption":
            # Data-related services show corruption
            data_services = ["api_order", "api_inventory", "api_user", "api_payment"]
            for service in random.sample(data_services, random.randint(2, 4)):
                sequence.append(self.generate_log_entry_string(
                    service, True, request_id, user_id,
                    random.choice(error_messages)
                ))
        
        elif anomaly_category == "concurrent_conflict":
            # Show retry pattern with conflicts
            primary_service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            # Conflict occurs
            sequence.append(self.generate_log_entry_string(
                primary_service, True, request_id, user_id,
                random.choice(error_messages)
            ))
            
            # Retry
            sequence.append(self.generate_log_entry_string(primary_service, False, request_id, user_id))
            
            # Conflict again
            sequence.append(self.generate_log_entry_string(
                primary_service, True, request_id, user_id,
                random.choice(error_messages)
            ))
        
        elif anomaly_category == "configuration_error":
            # Config service fails
            sequence.append(self.generate_log_entry_string(
                "api_config", True, request_id, user_id,
                random.choice(error_messages)
            ))
            
            # Services that depend on config also fail
            for service in ["api_inventory", "api_analytics"]:
                sequence.append(self.generate_log_entry_string(
                    service, True, request_id, user_id,
                    random.choice(error_messages)
                ))
        
        elif anomaly_category == "database_connection_failure":
            # Database-related services fail
            db_services = ["api_order", "api_user", "api_payment", "api_inventory"]
            for service in random.sample(db_services, random.randint(2, 4)):
                sequence.append(self.generate_log_entry_string(
                    service, True, request_id, user_id,
                    random.choice(error_messages)
                ))
        
        # Fill to required length
        while len(sequence) < sequence_length:
            service = random.choice(self.services)
            sequence.append(self.generate_log_entry_string(
                service, random.choice([True, False]), request_id, user_id
            ))
        
        return sequence
    
    def generate_classification_dataset(self, samples_per_category: int = 1000) -> List[Dict[str, Any]]:
        """Generate dataset for anomaly classification"""
        dataset = []
        
        # Generate normal sequences
        print(f"Generating {samples_per_category} normal sequences...")
        for i in range(samples_per_category):
            if i % 200 == 0:
                print(f"Normal progress: {i}/{samples_per_category}")
            sequence_length = random.randint(6, 15)
            log_strings = self.generate_normal_sequence(sequence_length)
            dataset.append({
                "sequence_id": str(uuid.uuid4()),
                "category": "normal",
                "label": "normal",
                "description": "Normal operation sequence",
                "length": len(log_strings),
                "log_entries": log_strings
            })
        
        # Generate anomaly sequences for each category
        for category in self.anomaly_categories.keys():
            print(f"Generating {samples_per_category} {category} sequences...")
            for i in range(samples_per_category):
                if i % 200 == 0:
                    print(f"{category} progress: {i}/{samples_per_category}")
                sequence_length = random.randint(6, 15)
                log_strings = self.generate_anomaly_sequence(category, sequence_length)
                dataset.append({
                    "sequence_id": str(uuid.uuid4()),
                    "category": category,
                    "label": category,
                    "description": self.anomaly_categories[category]["description"],
                    "length": len(log_strings),
                    "log_entries": log_strings
                })
        
        # Shuffle dataset
        random.shuffle(dataset)
        return dataset
    
    def save_classification_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save classification dataset"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        categories = {}
        for item in dataset:
            cat = item["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"Classification dataset saved to {filename}")
        print(f"Total sequences: {len(dataset)}")
        print("Category distribution:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")
    
    def generate_training_samples(self, dataset: List[Dict[str, Any]], num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate training samples in Unsloth instruction tuning format"""
        training_samples = []
        
        # Define allowed categories for constraint
        allowed_categories = [
            "normal", "service_cascade_failure", "authentication_failure", 
            "resource_exhaustion", "network_timeout", "data_corruption"
        ]
        
        for item in random.sample(dataset, min(num_samples, len(dataset))):
            # Format log entries as a single string
            log_text = "\n".join(item["log_entries"])
            
            # Determine if category is in allowed list
            category = item["category"]
            is_allowed = category in allowed_categories
            
            # Generate confidence score (higher for clear patterns, lower for ambiguous)
            if category == "normal":
                confidence = random.uniform(0.85, 0.98)  # Normal sequences are usually clear
            elif category in ["service_cascade_failure", "authentication_failure"]:
                confidence = random.uniform(0.80, 0.95)  # Clear patterns
            elif category in ["resource_exhaustion", "network_timeout"]:
                confidence = random.uniform(0.75, 0.90)  # Moderate patterns
            else:
                confidence = random.uniform(0.70, 0.85)  # Less clear patterns
            
            # Create instruction with category constraints
            instruction = """Please analyze the log sequence and classify it into one of the following categories:

ALLOWED CATEGORIES:
- normal: Normal operation sequence
- service_cascade_failure: A service failure that cascades to other dependent services
- authentication_failure: Authentication and authorization related failures
- resource_exhaustion: System resources are exhausted (memory, disk, connections)
- network_timeout: Network connectivity and timeout issues
- data_corruption: Data integrity and corruption issues

If the log sequence shows a different type of anomaly not in the allowed categories, respond with "UNKNOWN" and suggest the most likely category.

Provide your response in the following format:
**Classification**: [category]
**Confidence**: [0.0-1.0 score]
**Description**: [brief explanation]
**Key Indicators**: [specific log patterns that led to this classification]"""
            
            # Create output based on whether category is allowed
            if is_allowed:
                output = f"""**Classification**: {category}
**Confidence**: {confidence:.2f}
**Description**: {item['description']}
**Key Indicators**: The logs show {item['description'].lower()}"""
            else:
                # For unknown categories, suggest the most similar allowed category
                suggestions = {
                    "concurrent_conflict": "data_corruption",
                    "configuration_error": "authentication_failure", 
                    "database_connection_failure": "network_timeout"
                }
                suggested = suggestions.get(category, "service_cascade_failure")
                output = f"""**Classification**: UNKNOWN
**Confidence**: {confidence:.2f}
**Description**: This anomaly type is not in the allowed categories
**Key Indicators**: The logs show {item['description'].lower()}
**Suggested Category**: {suggested} (closest match to allowed categories)"""
            
            # Create instruction tuning format for Unsloth
            sample = {
                "instruction": instruction,
                "input": log_text,
                "output": output
            }
            training_samples.append(sample)
        
        return training_samples

def main():
    generator = AnomalyClassificationGenerator()
    
    print("Generating anomaly classification dataset...")
    print(f"Anomaly categories: {list(generator.anomaly_categories.keys())}")
    
    # Generate dataset
    dataset = generator.generate_classification_dataset(samples_per_category=500)
    
    # Save full dataset
    generator.save_classification_dataset(dataset, "data/anomaly_classification_dataset.json")
    
    # Generate training samples for Unsloth instruction tuning
    training_samples = generator.generate_training_samples(dataset, num_samples=2000)
    
    with open("data/unsloth_training_data.json", 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Unsloth instruction tuning data saved to data/unsloth_training_data.json")
    print(f"Training samples: {len(training_samples)}")
    
    # Show examples
    print("\nExample normal sequence:")
    normal_sequence = generator.generate_normal_sequence(6)
    for i, log in enumerate(normal_sequence):
        print(f"  {i+1}. {log}")
    
    print("\nExample anomaly sequence (service_cascade_failure):")
    anomaly_sequence = generator.generate_anomaly_sequence("service_cascade_failure", 6)
    for i, log in enumerate(anomaly_sequence):
        print(f"  {i+1}. {log}")

if __name__ == "__main__":
    main() 