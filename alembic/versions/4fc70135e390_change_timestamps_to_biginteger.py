"""Change timestamps to BigInteger

Revision ID: 4fc70135e390
Revises: c22371c14d32
Create Date: 2024-06-08 19:00:34.387211

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4fc70135e390'
down_revision: Union[str, None] = 'c22371c14d32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
