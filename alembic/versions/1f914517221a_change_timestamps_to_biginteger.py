from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '1f914517221a'
down_revision = '4fc70135e390'
branch_labels = None
depends_on = None

def upgrade():
    # Step 1: Add a new temporary column
    op.add_column('candlestick_images', sa.Column('timestamp_temp', sa.BIGINT(), nullable=True))
    op.add_column('data_frame_metadata', sa.Column('timestamp_temp', sa.BIGINT(), nullable=True))

    # Step 2: Update the new column with the converted timestamp values, assuming timestamps are in seconds and need to be converted to epoch
    op.execute('''
        UPDATE candlestick_images
        SET timestamp_temp = timestamp
    ''')
    op.execute('''
        UPDATE data_frame_metadata
        SET timestamp_temp = EXTRACT(EPOCH FROM timestamp::timestamp)::bigint
    ''')

    # Step 3: Drop the old timestamp column
    op.drop_column('candlestick_images', 'timestamp')
    op.drop_column('data_frame_metadata', 'timestamp')

    # Step 4: Rename the new column to timestamp
    op.alter_column('candlestick_images', 'timestamp_temp', new_column_name='timestamp')
    op.alter_column('data_frame_metadata', 'timestamp_temp', new_column_name='timestamp')

def downgrade():
    # Step 1: Add a new temporary column with timestamp type
    op.add_column('candlestick_images', sa.Column('timestamp_temp', sa.TIMESTAMP(), nullable=True))
    op.add_column('data_frame_metadata', sa.Column('timestamp_temp', sa.TIMESTAMP(), nullable=True))

    # Step 2: Update the new column with the converted BIGINT values
    op.execute('''
        UPDATE candlestick_images
        SET timestamp_temp = timestamp
    ''')
    op.execute('''
        UPDATE data_frame_metadata
        SET timestamp_temp = TO_TIMESTAMP(timestamp)
    ''')

    # Step 3: Drop the BIGINT timestamp column
    op.drop_column('candlestick_images', 'timestamp')
    op.drop_column('data_frame_metadata', 'timestamp')

    # Step 4: Rename the new column to timestamp
    op.alter_column('candlestick_images', 'timestamp_temp', new_column_name='timestamp')
    op.alter_column('data_frame_metadata', 'timestamp_temp', new_column_name='timestamp')
