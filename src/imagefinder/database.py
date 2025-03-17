#!/usr/bin/env python3

import os
import time
import sqlite3
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging
from imagefinder.image_types import ImageInfo

def init_database(db_path: str) -> sqlite3.Connection:
    """
    Initialize the database with required tables
    
    Args:
        db_path: Path to the database file
    
    Returns:
        SQLite database connection
    """
    # Create directory for the database if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to the database
    db_conn = sqlite3.connect(db_path)
    
    # Create the images table if it doesn't exist
    cursor = db_conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        source_prefix TEXT NOT NULL,
        format TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        modified_at TEXT NOT NULL,
        size INTEGER NOT NULL,
        average_hash TEXT NOT NULL,
        perceptual_hash TEXT NOT NULL,
        is_raw_format INTEGER NOT NULL,
        UNIQUE(path, source_prefix)
    )
    ''')
    
    # Create indexes for faster search
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_path ON images(path)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_prefix ON images(source_prefix)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_average_hash ON images(average_hash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON images(perceptual_hash)')
    
    db_conn.commit()
    return db_conn

def open_database(db_path: str) -> sqlite3.Connection:
    """
    Open an existing database
    
    Args:
        db_path: Path to the database file
    
    Returns:
        SQLite database connection
    
    Raises:
        Exception: If the database does not exist or cannot be opened
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        db_conn = sqlite3.connect(db_path)
        return db_conn
    except Exception as e:
        raise Exception(f"Failed to open database: {str(e)}")

def check_image_exists(db_conn: sqlite3.Connection, path: str, source_prefix: str) -> Tuple[bool, str]:
    """
    Check if an image already exists in the database
    
    Args:
        db_conn: Database connection
        path: Path to the image
        source_prefix: Source prefix
    
    Returns:
        Tuple of (exists, modified_at) where:
        - exists is a boolean indicating if the image exists
        - modified_at is the stored modification time as a string
    """
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT modified_at FROM images WHERE path = ? AND source_prefix = ?",
        (path, source_prefix)
    )
    
    row = cursor.fetchone()
    if row:
        return True, row[0]
    
    return False, ""

def store_image_info(db_conn: sqlite3.Connection, image_info: ImageInfo, force_rewrite: bool) -> None:
    """
    Store image information in the database
    
    Args:
        db_conn: Database connection
        image_info: Image information to store
        force_rewrite: Whether to force rewrite existing entries
    """
    cursor = db_conn.cursor()
    
    # Check if exists
    exists, _ = check_image_exists(db_conn, image_info.path, image_info.source_prefix)
    
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    if exists and force_rewrite:
        # Update existing record
        cursor.execute('''
        UPDATE images SET 
            format = ?,
            width = ?,
            height = ?,
            modified_at = ?,
            size = ?,
            average_hash = ?,
            perceptual_hash = ?,
            is_raw_format = ?
        WHERE path = ? AND source_prefix = ?
        ''', (
            image_info.format,
            image_info.width,
            image_info.height,
            image_info.modified_at,
            image_info.size,
            image_info.average_hash,
            image_info.perceptual_hash,
            1 if image_info.is_raw_format else 0,
            image_info.path,
            image_info.source_prefix
        ))
    elif not exists:
        # Insert new record
        cursor.execute('''
        INSERT INTO images (
            path, source_prefix, format, width, height, created_at, modified_at,
            size, average_hash, perceptual_hash, is_raw_format
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_info.path,
            image_info.source_prefix,
            image_info.format,
            image_info.width,
            image_info.height,
            current_time,
            image_info.modified_at,
            image_info.size,
            image_info.average_hash,
            image_info.perceptual_hash,
            1 if image_info.is_raw_format else 0
        ))
    
    db_conn.commit()

# Aliases for better compatibility with original Go code
InitDatabase = init_database
OpenDatabase = open_database
CheckImageExists = check_image_exists
StoreImageInfo = store_image_info