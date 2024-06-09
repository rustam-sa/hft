"""Add new models

Revision ID: 0fe3a3979950
Revises: 0c2751aa67e8
Create Date: 2024-06-08 17:39:18.481249

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0fe3a3979950'
down_revision: Union[str, None] = '0c2751aa67e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass

def downgrade() -> None:
    pass
