"""Change timestamps to BigInteger

Revision ID: c22371c14d32
Revises: d40ef8038beb
Create Date: 2024-06-08 18:59:19.077936

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c22371c14d32'
down_revision: Union[str, None] = 'd40ef8038beb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
