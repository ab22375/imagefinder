#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

@dataclass
class ImageInfo:
    """Holds the image metadata and features"""
    id: int
    path: str
    source_prefix: str
    format: str
    width: int
    height: int
    created_at: str
    modified_at: str
    size: int
    average_hash: str
    perceptual_hash: str
    is_raw_format: bool
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "path": self.path,
            "source_prefix": self.source_prefix,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "size": self.size,
            "average_hash": self.average_hash,
            "perceptual_hash": self.perceptual_hash,
            "is_raw_format": self.is_raw_format
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary"""
        return cls(
            id=data.get("id", 0),
            path=data.get("path", ""),
            source_prefix=data.get("source_prefix", ""),
            format=data.get("format", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            size=data.get("size", 0),
            average_hash=data.get("average_hash", ""),
            perceptual_hash=data.get("perceptual_hash", ""),
            is_raw_format=data.get("is_raw_format", False)
        )


@dataclass
class ImageMatch:
    """Holds the similarity scores"""
    path: str
    source_prefix: str
    ssim_score: float