"""Add new models

Revision ID: d40ef8038beb
Revises: 0fe3a3979950
Create Date: 2024-06-08 17:56:16.355282

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd40ef8038beb'
down_revision: Union[str, None] = '0fe3a3979950'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
