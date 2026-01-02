import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { PaymentType } from '../enums/payment-type.enum';
import { PaymentStatus } from '../enums/payment-status.enum';
import { PaymentDirection } from '../enums/payment-direction.enum';

export class PaymentResponseDto {
  @ApiProperty({
    description: 'Payment ID (UUID)',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  id: string;

  @ApiProperty({
    description: 'External reference',
    example: 'INV-2024-001',
  })
  externalRef: string;

  @ApiPropertyOptional({
    description: 'JPMorgan payment ID',
    example: 'JPM-12345678',
  })
  jpmPaymentId?: string;

  @ApiProperty({
    description: 'Payment type',
    enum: PaymentType,
    example: PaymentType.ACH,
  })
  type: PaymentType;

  @ApiProperty({
    description: 'Payment direction',
    enum: PaymentDirection,
    example: PaymentDirection.CREDIT,
  })
  direction: PaymentDirection;

  @ApiProperty({
    description: 'Payment status',
    enum: PaymentStatus,
    example: PaymentStatus.CREATED,
  })
  status: PaymentStatus;

  @ApiProperty({
    description: 'Amount in cents',
    example: 100000,
  })
  amountCents: number;

  @ApiProperty({
    description: 'Amount in dollars',
    example: 1000.00,
  })
  amountDollars: number;

  @ApiProperty({
    description: 'Currency code',
    example: 'USD',
  })
  currency: string;

  @ApiPropertyOptional({
    description: 'Source account ID',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  fromAccountId?: string;

  @ApiProperty({
    description: 'Destination account number',
    example: '1234567890',
  })
  toAccountNumber: string;

  @ApiProperty({
    description: 'Destination routing number',
    example: '021000021',
  })
  toRoutingNumber: string;

  @ApiProperty({
    description: 'Destination account name',
    example: 'John Doe',
  })
  toAccountName: string;

  @ApiPropertyOptional({
    description: 'Destination bank name',
    example: 'JPMorgan Chase Bank',
  })
  toBankName?: string;

  @ApiPropertyOptional({
    description: 'Payment description',
    example: 'Payment for Invoice INV-2024-001',
  })
  description?: string;

  @ApiPropertyOptional({
    description: 'Reference number',
    example: 'INV-2024-001',
  })
  referenceNumber?: string;

  @ApiProperty({
    description: 'Organization ID',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  organizationId: string;

  @ApiProperty({
    description: 'Created by user ID',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  createdById: string;

  @ApiPropertyOptional({
    description: 'Approved by user ID',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  approvedById?: string;

  @ApiProperty({
    description: 'Created at timestamp',
    example: '2024-01-02T08:00:00.000Z',
  })
  createdAt: Date;

  @ApiProperty({
    description: 'Updated at timestamp',
    example: '2024-01-02T08:30:00.000Z',
  })
  updatedAt: Date;

  @ApiPropertyOptional({
    description: 'Approved at timestamp',
    example: '2024-01-02T08:15:00.000Z',
  })
  approvedAt?: Date;

  @ApiPropertyOptional({
    description: 'Submitted at timestamp',
    example: '2024-01-02T08:20:00.000Z',
  })
  submittedAt?: Date;

  @ApiPropertyOptional({
    description: 'Settled at timestamp',
    example: '2024-01-02T10:00:00.000Z',
  })
  settledAt?: Date;

  @ApiPropertyOptional({
    description: 'Failed at timestamp',
    example: '2024-01-02T08:25:00.000Z',
  })
  failedAt?: Date;

  @ApiPropertyOptional({
    description: 'Error message if payment failed',
    example: 'Insufficient funds',
  })
  errorMessage?: string;

  @ApiPropertyOptional({
    description: 'Error code if payment failed',
    example: 'INSUFFICIENT_FUNDS',
  })
  errorCode?: string;

  @ApiPropertyOptional({
    description: 'Additional metadata',
    example: { invoiceId: '12345', customerId: '67890' },
  })
  metadata?: any;

  @ApiProperty({
    description: 'Whether payment is in terminal state',
    example: false,
  })
  isTerminal: boolean;

  @ApiProperty({
    description: 'Whether payment can be approved',
    example: true,
  })
  canBeApproved: boolean;

  @ApiProperty({
    description: 'Whether payment can be submitted',
    example: false,
  })
  canBeSubmitted: boolean;
}

export class PaymentListResponseDto {
  @ApiProperty({
    description: 'List of payments',
    type: [PaymentResponseDto],
  })
  data: PaymentResponseDto[];

  @ApiProperty({
    description: 'Total count of payments',
    example: 100,
  })
  total: number;

  @ApiProperty({
    description: 'Current page',
    example: 1,
  })
  page: number;

  @ApiProperty({
    description: 'Page size',
    example: 20,
  })
  pageSize: number;

  @ApiProperty({
    description: 'Total pages',
    example: 5,
  })
  totalPages: number;
}
