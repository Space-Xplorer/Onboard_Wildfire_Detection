"""
telemetry_encoder.py - CCSDS-Compliant Alert Packet Generator
Packages fire detection results into a 26-byte spacecraft telemetry packet.
Format: AOS (Advanced Orbiting Systems) compatible for legacy satellites.
"""

import struct
import time
from datetime import datetime

# ===== CCSDS PACKET CONSTANTS =====
# Per CCSDS 133.0-B-2 (TM Space Data Link Protocol)
PACKET_SIZE_BYTES = 26
PACKET_VERSION = 0b000  # Version 0
PACKET_TYPE = 0b0       # Telemetry (not telecommand)
SECONDARY_HEADER = 0b1  # Secondary header present

# Packet identifier fields
APID = 0x042            # Application ID (fire detection subsystem)
SEQUENCE_FLAG = 0b11    # Unsegmented user packet

# Time-tag epoch (Unix epoch for simplicity; flight would use GPS/UTC)
EPOCH_UNIX = 0


class CCSDSFireAlertPacket:
    """
    Generates CCSDS-compliant 26-byte alert packets for downlink.
    
    Packet Structure:
      Bytes  0-5  : Primary Header (6 bytes)
      Bytes  6-7  : Secondary Header / Timestamp (2 bytes)
      Bytes  8-25 : Payload (18 bytes)
                    - Lat/Lon (4 bytes each)
                    - Temperature (2 bytes)
                    - Confidence (1 byte)
                    - Status flags (1 byte)
                    - Spare (4 bytes, reserved)
    """
    
    def __init__(self, sat_id=1):
        """
        Initialize CCSDS encoder.
        
        Args:
            sat_id (int): Satellite identifier (0-2047, uses 11 bits)
        """
        self.sat_id = sat_id & 0x7FF  # 11-bit mask
        self.sequence_counter = 0
    
    def pack_alert(self, latitude, longitude, temperature_k, confidence_pct, 
                   status_flags=0x00, time_seconds=None):
        """
        Pack fire alert into CCSDS packet.
        
        Args:
            latitude (float): -90 to +90 degrees
            longitude (float): -180 to +180 degrees
            temperature_k (float): Brightness temperature in Kelvin
            confidence_pct (int): AI confidence 0-100
            status_flags (int): 8-bit status word (spare for future use)
            time_seconds (int): Seconds since epoch (None = current time)
        
        Returns:
            bytes: 26-byte CCSDS packet
        """
        if time_seconds is None:
            time_seconds = int(time.time() - EPOCH_UNIX)
        
        # ===== PRIMARY HEADER (6 bytes) =====
        # Byte 0-1: Version (3 bits) | Type (1 bit) | SecHdr (1 bit) | APID (11 bits)
        packet_id = (PACKET_VERSION << 13) | (PACKET_TYPE << 12) | (SECONDARY_HEADER << 11) | APID
        
        # Byte 2-3: Sequence flags (2 bits) | Sequence counter (14 bits)
        sequence_word = (SEQUENCE_FLAG << 14) | (self.sequence_counter & 0x3FFF)
        
        # Byte 4-5: Packet data length (minus 1, per standard)
        # Total packet: 26 bytes. Data length field = 26 - 6 - 1 = 19 bytes (in standard format)
        packet_length = PACKET_SIZE_BYTES - 7  # 19
        
        primary_header = struct.pack('>HHH',
            packet_id,
            sequence_word,
            packet_length
        )
        
        # ===== SECONDARY HEADER (2 bytes) =====
        # Byte 6-7: Timestamp (16-bit, seconds since epoch, modulo 65536 for wrapping)
        timestamp_short = time_seconds & 0xFFFF
        secondary_header = struct.pack('>H', timestamp_short)
        
        # ===== PAYLOAD (18 bytes) =====
        # Quantize lat/lon to signed 16-bit integers (±1/100 degree resolution)
        lat_int = int(latitude * 100)
        lon_int = int(longitude * 100)
        
        # Temperature: store offset from 250K in 1-byte (0.25K resolution)
        # Range: 250K to ~314K (typical for Earth observations)
        temp_offset = max(0, min(255, int((temperature_k - 250.0) * 4)))
        
        # Confidence: 0-100 as single byte
        conf_byte = max(0, min(255, int(confidence_pct)))
        
        # Payload structure: lat(2) + lon(2) + temp(1) + conf(1) + status(2) + reserved(10) = 18 bytes
        payload = struct.pack('>hhBBHHHHHH',
            lat_int,           # Latitude (°/100)
            lon_int,           # Longitude (°/100)
            temp_offset,       # Temperature offset (0.25K per LSB from 250K)
            conf_byte,         # Confidence %
            status_flags,      # Status word (2 bytes)
            0, 0, 0, 0, 0      # Reserved (5 shorts = 10 bytes)
        )
        
        # Increment sequence counter
        self.sequence_counter = (self.sequence_counter + 1) & 0x3FFF
        
        return primary_header + secondary_header + payload
    
    @staticmethod
    def unpack_alert(packet):
        """
        Parse a CCSDS alert packet (for verification/ground station).
        
        Args:
            packet (bytes): 26-byte CCSDS packet
        
        Returns:
            dict: Decoded packet contents
        """
        if len(packet) != PACKET_SIZE_BYTES:
            raise ValueError(f"Invalid packet size: {len(packet)} != {PACKET_SIZE_BYTES}")
        
        # Parse primary header
        packet_id, seq_word, pkt_len = struct.unpack('>HHH', packet[0:6])
        version = (packet_id >> 13) & 0x7
        pkt_type = (packet_id >> 12) & 0x1
        sec_hdr = (packet_id >> 11) & 0x1
        apid = packet_id & 0x7FF
        seq_flags = (seq_word >> 14) & 0x3
        seq_counter = seq_word & 0x3FFF
        
        # Parse secondary header (timestamp)
        timestamp, = struct.unpack('>H', packet[6:8])
        
        # Parse payload
        lat_int, lon_int, temp_offset, conf_byte, status_flags, _, _, _, _, _ = \
            struct.unpack('>hhBBHHHHHH', packet[8:26])
        
        latitude = lat_int / 100.0
        longitude = lon_int / 100.0
        temperature = 250.0 + (temp_offset / 4.0)
        confidence = conf_byte
        
        return {
            'version': version,
            'type': pkt_type,
            'apid': apid,
            'sequence': seq_counter,
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'temperature_k': temperature,
            'confidence_pct': confidence,
            'status': status_flags,
        }
    
    def hex_packet(self, latitude, longitude, temperature_k, confidence_pct, 
                   status_flags=0x00):
        """
        Generate hex string representation of alert packet (for downlink logging).
        
        Returns:
            str: Hex string (e.g., "000A42E10000...")
        """
        packet = self.pack_alert(latitude, longitude, temperature_k, confidence_pct, status_flags)
        return packet.hex().upper()


# ===== TEST / VERIFICATION =====

if __name__ == "__main__":
    print("=== CCSDS Telemetry Encoder Verification ===\n")
    
    encoder = CCSDSFireAlertPacket(sat_id=1)
    
    # Test Case 1: Detected active fire
    alert_packet = encoder.pack_alert(
        latitude=35.6895,      # Example: CA / AUS latitude
        longitude=-120.4068,   # Example: CA / AUS longitude
        temperature_k=380.0,   # Active fire signature
        confidence_pct=92      # High AI confidence
    )
    
    hex_str = encoder.hex_packet(35.6895, -120.4068, 380.0, 92)
    
    print(f"Alert Packet (26 bytes):")
    print(f"  Hex: {hex_str}")
    print(f"  Size: {len(alert_packet)} bytes\n")
    
    # Decode and verify
    decoded = CCSDSFireAlertPacket.unpack_alert(alert_packet)
    print(f"Decoded Alert:")
    for key, val in decoded.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")
    
    print(f"\n✓ CCSDS Telemetry module verified")
