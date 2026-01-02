import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  OneToMany,
  JoinColumn,
  Index,
} from 'typeorm';
import { PaymentType } from '../enums/payment-type.enum';
import { PaymentStatus } from '../enums/payment-status.enum';
import { PaymentDirection } from '../enums/payment-direction.enum';
import { User } from '../../users/user.entity';
import { Organization } from '../../organizations/organization.entity';
import { BankAccount } from '../../accounts/bank-account.entity';

@Entity('payments')
@Index(['organizationId', 'status'])
@Index(['type', 'status'])
@Index(['createdAt'])
@Index(['externalRef'], { unique: true })
export class Payment {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  // External reference (your system's reference)
  @Column({ unique: true, nullable: false })
  externalRef: string;

  // JPMorgan payment ID (from their API)
  @Column({ nullable: true })
  jpmPaymentId: string;

  // Payment type (ACH, WIRE, RTP)
  @Column({
    type: 'enum',
    enum: PaymentType,
    nullable: false,
  })
  type: PaymentType;

  // Payment direction (CREDIT = outgoing, DEBIT = incoming)
  @Column({
    type: 'enum',
    enum: PaymentDirection,
    nullable: false,
  })
  direction: PaymentDirection;

  // Payment status
  @Column({
    type: 'enum',
    enum: PaymentStatus,
    default: PaymentStatus.CREATED,
  })
  status: PaymentStatus;

  // Amount in cents (to avoid floating point issues)
  @Column({ type: 'bigint', nullable: false })
  amountCents: number;

  // Currency code (ISO 4217)
  @Column({ length: 3, default: 'USD' })
  currency: string;

  // Source account
  @ManyToOne(() => BankAccount, { nullable: true })
  @JoinColumn({ name: 'fromAccountId' })
  fromAccount: BankAccount;

  @Column({ nullable: true })
  fromAccountId: string;

  // Destination account details
  @Column({ nullable: true })
  toAccountNumber: string;

  @Column({ nullable: true })
  toRoutingNumber: string;

  @Column({ nullable: true })
  toAccountName: string;

  @Column({ nullable: true })
  toBankName: string;

  // Payment description/memo
  @Column({ type: 'text', nullable: true })
  description: string;

  // Reference/invoice number
  @Column({ nullable: true })
  referenceNumber: string;

  // Organization
  @ManyToOne(() => Organization, { nullable: false })
  @JoinColumn({ name: 'organizationId' })
  organization: Organization;

  @Column({ nullable: false })
  organizationId: string;

  // User who created the payment
  @ManyToOne(() => User, { nullable: false })
  @JoinColumn({ name: 'createdById' })
  createdBy: User;

  @Column({ nullable: false })
  createdById: string;

  // User who approved the payment
  @ManyToOne(() => User, { nullable: true })
  @JoinColumn({ name: 'approvedById' })
  approvedBy: User;

  @Column({ nullable: true })
  approvedById: string;

  // Timestamps
  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  approvedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  submittedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  settledAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  failedAt: Date;

  // Idempotency key for preventing duplicate submissions
  @Column({ unique: true, nullable: false })
  idempotencyKey: string;

  // Raw request/response from JPMorgan (for debugging)
  @Column({ type: 'jsonb', nullable: true })
  rawRequest: any;

  @Column({ type: 'jsonb', nullable: true })
  rawResponse: any;

  // Error details if payment failed
  @Column({ type: 'text', nullable: true })
  errorMessage: string;

  @Column({ type: 'text', nullable: true })
  errorCode: string;

  // Metadata (flexible JSON field for additional data)
  @Column({ type: 'jsonb', nullable: true })
  metadata: any;

  // Soft delete
  @Column({ type: 'timestamp', nullable: true })
  deletedAt: Date;

  // Helper method to get amount in dollars
  getAmountDollars(): number {
    return this.amountCents / 100;
  }

  // Helper method to check if payment is terminal
  isTerminal(): boolean {
    return [
      PaymentStatus.COMPLETED,
      PaymentStatus.FAILED,
      PaymentStatus.CANCELLED,
      PaymentStatus.RETURNED,
    ].includes(this.status);
  }

  // Helper method to check if payment can be approved
  canBeApproved(): boolean {
    return [
      PaymentStatus.CREATED,
      PaymentStatus.PENDING_APPROVAL,
    ].includes(this.status);
  }

  // Helper method to check if payment can be submitted
  canBeSubmitted(): boolean {
    return [
      PaymentStatus.APPROVED,
      PaymentStatus.READY_TO_SUBMIT,
    ].includes(this.status);
  }
}
