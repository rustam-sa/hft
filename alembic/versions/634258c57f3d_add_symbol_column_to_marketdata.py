"""Add symbol column to MarketData

Revision ID: 634258c57f3d
Revises: b4ac8040f9f9
Create Date: 2024-06-01 11:24:53.255474

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '634258c57f3d'
down_revision: Union[str, None] = 'b4ac8040f9f9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('market_data', sa.Column('symbol', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('market_data', 'symbol')
    # ### end Alembic commands ###
