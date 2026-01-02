# 🚀 HYBRID APPROACH - PHASE 1 IMPLEMENTATION PLAN

**Selected Path:** Option 4 - Hybrid Approach  
**Current Phase:** Phase 1 - Complete Sandbox Payment System  
**Timeline:** Weeks 1-3 (Starting Immediately)  
**Status:** ✅ APPROVED - READY TO START

---

## 📊 PHASE 1 OVERVIEW

### **Objective**
Build complete ACH, Wire, and RTP payment modules in sandbox for immediate testing and development.

### **Timeline**
- **Week 1:** Complete ACH Module (40 hours)
- **Week 2:** Complete Wire & RTP Modules (40 hours)
- **Week 3:** Enhanced Dashboard & Testing (16 hours)
- **Total:** 96 hours over 3 weeks

### **Team Requirements**
- **Option A:** 1 senior developer (full-time for 3 weeks)
- **Option B:** 2 developers (part-time, 20 hours/week each)

### **Deliverables**
✅ Complete ACH payment module (12 files)  
✅ Complete Wire transfer module (12 files)  
✅ Complete RTP payment module (14 files)  
✅ Approval workflow system (10 files)  
✅ Enhanced live transaction dashboard  
✅ Full test coverage  
✅ Documentation

---

## 📅 WEEK 1: COMPLETE ACH MODULE

### **Day 1: ACH Validation & NACHA Generation (8 hours)**

#### **Task 1.1: Create ACH Validation Service**
**File:** `nestjs-backend/src/ach/services/ach-validation.service.ts`

**Implementation:**
```typescript
import { Injectable, BadRequestException } from '@nestjs/common';
import { CreateAchDto } from '../dtos/create-ach.dto';

@Injectable()
export class AchValidationService {
  /**
   * Validate routing number using ABA checksum algorithm
   */
  validateRoutingNumber(routingNumber: string): boolean {
    if (!/^\d{9}$/.test(routingNumber)) {
      throw new BadRequestException('Routing number must be 9 digits');
    }

    // ABA checksum validation
    const digits = routingNumber.split('').map(Number);
    const checksum = 
      (3 * (digits[0] + digits[3] + digits[6])) +
      (7 * (digits[1] + digits[4] + digits[7])) +
      (1 * (digits[2] + digits[5] + digits[8]));

    if (checksum % 10 !== 0) {
      throw new BadRequestException('Invalid routing number checksum');
    }

    return true;
  }

  /**
   * Validate account number format
   */
  validateAccountNumber(accountNumber: string): boolean {
    if (!/^\d{1,17}$/.test(accountNumber)) {
      throw new BadRequestException('Account number must be 1-17 digits');
    }
    return true;
  }

  /**
   * Validate ACH amount limits
   */
  validateAmount(amountCents: number, sameDayAch: boolean): boolean {
    const maxAmount = sameDayAch ? 100000000 : 1000000000; // $1M for same-day, $10M for standard
    
    if (amountCents < 1) {
      throw new BadRequestException('Amount must be at least $0.01');
    }

    if (amountCents > maxAmount) {
      const maxDollars = maxAmount / 100;
      throw new BadRequestException(`Amount exceeds maximum of $${maxDollars.toLocaleString()}`);
    }

    return true;
  }

  /**
   * Validate complete ACH payment
   */
  async validateAchPayment(dto: CreateAchDto): Promise<void> {
    // Validate routing number
    this.validateRoutingNumber(dto.receiverRoutingNumber);

    // Validate account number
    this.validateAccountNumber(dto.receiverAccountNumber);

    // Validate amount
    this.validateAmount(dto.amountCents, dto.sameDayAch || false);

    // Validate effective date (if provided)
    if (dto.effectiveDate) {
      const effectiveDate = new Date(dto.effectiveDate);
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      if (effectiveDate < today) {
        throw new BadRequestException('Effective date cannot be in the past');
      }

      // Max 2 business days in future for same-day ACH
      if (dto.sameDayAch) {
        const maxDate = new Date(today);
        maxDate.setDate(maxDate.getDate() + 2);
        
        if (effectiveDate > maxDate) {
          throw new BadRequestException('Same-day ACH effective date cannot be more than 2 days in future');
        }
      }
    }

    // Validate originator ID (10 digits)
    if (!/^\d{10}$/.test(dto.originatorId)) {
      throw new BadRequestException('Originator ID must be 10 digits');
    }
  }
}
```

**Time:** 4 hours

---

#### **Task 1.2: Create NACHA File Generator**
**File:** `nestjs-backend/src/ach/services/nacha-generator.service.ts`

**Implementation:**
```typescript
import { Injectable } from '@nestjs/common';
import { AchPayment } from '../entities/ach-payment.entity';
import { Payment } from '../../payments-core/entities/payment.entity';

@Injectable()
export class NachaGeneratorService {
  /**
   * Generate NACHA file for ACH payment
   */
  generateNachaFile(payment: Payment, achPayment: AchPayment): string {
    const lines: string[] = [];

    // File Header Record (Type 1)
    lines.push(this.generateFileHeader());

    // Batch Header Record (Type 5)
    lines.push(this.generateBatchHeader(achPayment));

    // Entry Detail Record (Type 6)
    lines.push(this.generateEntryDetail(payment, achPayment));

    // Addenda Record (Type 7) - if addenda exists
    if (achPayment.addendaRecord) {
      lines.push(this.generateAddenda(achPayment));
    }

    // Batch Control Record (Type 8)
    lines.push(this.generateBatchControl(payment, achPayment));

    // File Control Record (Type 9)
    lines.push(this.generateFileControl(payment));

    return lines.join('\n');
  }

  private generateFileHeader(): string {
    const now = new Date();
    const fileCreationDate = this.formatDate(now, 'YYMMDD');
    const fileCreationTime = this.formatTime(now);

    return [
      '1',                          // Record Type Code
      '01',                         // Priority Code
      ' 021000021',                 // Immediate Destination (routing)
      ' 1234567890',                // Immediate Origin (company ID)
      fileCreationDate,             // File Creation Date
      fileCreationTime,             // File Creation Time
      'A',                          // File ID Modifier
      '094',                        // Record Size
      '10',                         // Blocking Factor
      '1',                          // Format Code
      'JPMORGAN CHASE'.padEnd(23),  // Destination Name
      'YOUR COMPANY'.padEnd(23),    // Origin Name
      '        ',                   // Reference Code
    ].join('');
  }

  private generateBatchHeader(achPayment: AchPayment): string {
    const now = new Date();
    const effectiveDate = achPayment.effectiveDate || now;

    return [
      '5',                                    // Record Type Code
      achPayment.transactionType === 'CREDIT' ? '220' : '225', // Service Class Code
      achPayment.originatorName.padEnd(16),   // Company Name
      ''.padEnd(20),                          // Company Discretionary Data
      achPayment.originatorId,                // Company Identification
      achPayment.secCode,                     // SEC Code
      'PAYMENT'.padEnd(10),                   // Entry Description
      ''.padEnd(6),                           // Descriptive Date
      this.formatDate(effectiveDate, 'YYMMDD'), // Effective Entry Date
      '   ',                                  // Settlement Date
      '1',                                    // Originator Status Code
      '02100002',                             // Originating DFI (first 8 of routing)
      '0000001',                              // Batch Number
    ].join('');
  }

  private generateEntryDetail(payment: Payment, achPayment: AchPayment): string {
    const transactionCode = this.getTransactionCode(achPayment);
    const amount = payment.amountCents.toString().padStart(10, '0');

    return [
      '6',                                      // Record Type Code
      transactionCode,                          // Transaction Code
      achPayment.receiverRoutingNumber.substring(0, 8), // Receiving DFI
      achPayment.receiverRoutingNumber.charAt(8), // Check Digit
      achPayment.receiverAccountNumber.padEnd(17), // DFI Account Number
      amount,                                   // Amount
      ''.padEnd(15),                            // Individual ID Number
      achPayment.receiverName.padEnd(22),       // Individual Name
      '  ',                                     // Discretionary Data
      achPayment.addendaRecord ? '1' : '0',     // Addenda Record Indicator
      achPayment.traceNumber || '000000000000000', // Trace Number
    ].join('');
  }

  private generateAddenda(achPayment: AchPayment): string {
    return [
      '7',                                      // Record Type Code
      '05',                                     // Addenda Type Code
      achPayment.addendaRecord.substring(0, 80).padEnd(80), // Payment Related Information
      '0001',                                   // Addenda Sequence Number
      '000000000000000',                        // Entry Detail Sequence Number
    ].join('');
  }

  private generateBatchControl(payment: Payment, achPayment: AchPayment): string {
    const entryCount = '000001';
    const entryHash = achPayment.receiverRoutingNumber.substring(0, 8);
    const totalDebit = achPayment.transactionType === 'DEBIT' ? payment.amountCents : 0;
    const totalCredit = achPayment.transactionType === 'CREDIT' ? payment.amountCents : 0;

    return [
      '8',                                      // Record Type Code
      achPayment.transactionType === 'CREDIT' ? '220' : '225', // Service Class Code
      entryCount,                               // Entry/Addenda Count
      entryHash.padStart(10, '0'),              // Entry Hash
      totalDebit.toString().padStart(12, '0'),  // Total Debit Entry Dollar Amount
      totalCredit.toString().padStart(12, '0'), // Total Credit Entry Dollar Amount
      achPayment.originatorId,                  // Company Identification
      ''.padEnd(19),                            // Message Authentication Code
      ''.padEnd(6),                             // Reserved
      '02100002',                               // Originating DFI
      '0000001',                                // Batch Number
    ].join('');
  }

  private generateFileControl(payment: Payment): string {
    return [
      '9',                                      // Record Type Code
      '000001',                                 // Batch Count
      '000001',                                 // Block Count
      '000001',                                 // Entry/Addenda Count
      '021000021'.padStart(10, '0'),            // Entry Hash
      payment.amountCents.toString().padStart(12, '0'), // Total Debit
      payment.amountCents.toString().padStart(12, '0'), // Total Credit
      ''.padEnd(39),                            // Reserved
    ].join('');
  }

  private getTransactionCode(achPayment: AchPayment): string {
    // Transaction codes:
    // 22 = Checking Credit, 23 = Checking Prenote Credit
    // 27 = Checking Debit, 28 = Checking Prenote Debit
    // 32 = Savings Credit, 33 = Savings Prenote Credit
    // 37 = Savings Debit, 38 = Savings Prenote Debit
    
    if (achPayment.transactionType === 'CREDIT') {
      return '22'; // Checking Credit (most common)
    } else {
      return '27'; // Checking Debit
    }
  }

  private formatDate(date: Date, format: string): string {
    const year = date.getFullYear().toString().substring(2);
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');

    if (format === 'YYMMDD') {
      return year + month + day;
    }
    return '';
  }

  private formatTime(date: Date): string {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return hours + minutes;
  }
}
```

**Time:** 4 hours

---

### **Day 2: ACH JP Morgan Client (8 hours)**

#### **Task 2.1: Create ACH JP Morgan Client**
**File:** `nestjs-backend/src/ach/services/ach-jpmorgan.client.ts`

**Implementation:**
```typescript
import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { JpmorganTokenService } from '../../connectors/jpmorgan/jpmorgan-token.service';
import { Payment } from '../../payments-core/entities/payment.entity';
import { AchPayment } from '../entities/ach-payment.entity';

export interface AchSubmissionResult {
  jpmPaymentId: string;
  traceNumber: string;
  status: string;
  submittedAt: Date;
}

@Injectable()
export class AchJpmorganClient {
  private readonly logger = new Logger(AchJpmorganClient.name);
  private readonly baseUrl: string;

  constructor(
    private readonly config: ConfigService,
    private readonly http: HttpService,
    private readonly tokenService: JpmorganTokenService,
  ) {
    this.baseUrl = this.config.get<string>('JPM_API_BASE_URL') || 
                   'https://api-sandbox.payments.jpmorgan.com';
  }

  /**
   * Submit ACH payment to JPMorgan
   */
  async submitAchPayment(
    payment: Payment,
    achPayment: AchPayment,
  ): Promise<AchSubmissionResult> {
    this.logger.log(`Submitting ACH payment ${payment.id} to JPMorgan`);

    try {
      const token = await this.tokenService.getAccessToken();
      const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'X-Idempotency-Key': payment.idempotencyKey || payment.id,
      };

      const payload = this.buildAchPayload(payment, achPayment);
      const url = `${this.baseUrl}/payments/v1/ach`;

      const response = await firstValueFrom(
        this.http.post(url, payload, { headers }),
      );

      this.logger.log(`ACH payment ${payment.id} submitted successfully`);

      return {
        jpmPaymentId: response.data.paymentId,
        traceNumber: response.data.traceNumber,
        status: response.data.status,
        submittedAt: new Date(),
      };
    } catch (error) {
      this.logger.error(`Failed to submit ACH payment ${payment.id}`, error);
      throw new Error(`JPMorgan ACH submission failed: ${error.message}`);
    }
  }

  /**
   * Get ACH payment status from JPMorgan
   */
  async getAchPaymentStatus(jpmPaymentId: string): Promise<any> {
    this.logger.log(`Fetching ACH payment status for ${jpmPaymentId}`);

    try {
      const token = await this.tokenService.getAccessToken();
      const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      };

      const url = `${this.baseUrl}/payments/v1/ach/${jpmPaymentId}`;

      const response = await firstValueFrom(
        this.http.get(url, { headers }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch ACH payment status for ${jpmPaymentId}`, error);
      throw new Error(`JPMorgan ACH status fetch failed: ${error.message}`);
    }
  }

  /**
   * Build ACH payload for JPMorgan API
   */
  private buildAchPayload(payment: Payment, achPayment: AchPayment): any {
    return {
      secCode: achPayment.secCode,
      transactionType: achPayment.transactionType,
      amount: {
        value: payment.amountCents / 100,
        currency: payment.currency,
      },
      originator: {
        name: achPayment.originatorName,
        id: achPayment.originatorId,
      },
      receiver: {
        name: achPayment.receiverName,
        accountNumber: achPayment.receiverAccountNumber,
        routingNumber: achPayment.receiverRoutingNumber,
      },
      effectiveDate: achPayment.effectiveDate?.toISOString().split('T')[0],
      sameDayAch: achPayment.sameDayAch,
      addenda: achPayment.addendaRecord,
      externalReference: payment.externalReference,
    };
  }
}
```

**Time:** 8 hours

---

### **Day 3-4: ACH Service & Controllers (16 hours)**

#### **Task 3.1: Complete ACH Service**
**File:** `nestjs-backend/src/ach/services/ach.service.ts`

*Already provided in PRODUCTION_PAYMENT_SYSTEM_PLAN.md - implement as specified*

**Time:** 8 hours

---

#### **Task 3.2: Create ACH Controller**
**File:** `nestjs-backend/src/ach/controllers/ach.controller.ts`

**Implementation:**
```typescript
import { Controller, Post, Get, Body, Param, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { AchService } from '../services/ach.service';
import { CreateAchDto } from '../dtos/create-ach.dto';
import { ApiKeyGuard } from '../../auth/api-key.guard';
import { RequireRole } from '../../auth/auth.decorator';
import { Role } from '../../auth/roles.enum';

@ApiTags('ACH Payments')
@Controller('api/ach')
@UseGuards(ApiKeyGuard)
@ApiBearerAuth()
export class AchController {
  constructor(private readonly achService: AchService) {}

  @Post('payments')
  @RequireRole(Role.ADMIN, Role.MAKER)
  @ApiOperation({ summary: 'Create ACH payment' })
  @ApiResponse({ status: 201, description: 'ACH payment created successfully' })
  async createAchPayment(
    @Body() dto: CreateAchDto,
    @Query('organizationId') organizationId: string,
    @Query('userId') userId: string,
  ) {
    const result = await this.achService.createAchPayment(dto, organizationId, userId);
    return {
      success: true,
      data: {
        paymentId: result.payment.id,
        achPaymentId: result.achPayment.id,
        status: result.payment.status,
        amount: result.payment.amountCents / 100,
        currency: result.payment.currency,
      },
    };
  }

  @Get('payments/:id')
  @RequireRole(Role.ADMIN, Role.VIEWER)
  @ApiOperation({ summary: 'Get ACH payment by ID' })
  async getAchPayment(@Param('id') id: string) {
    const result = await this.achService.getAchPayment(id);
    return {
      success: true,
      data: {
        payment: result.payment,
        achPayment: result.achPayment,
      },
    };
  }

  @Post('payments/:id/submit')
  @RequireRole(Role.ADMIN, Role.CHECKER)
  @ApiOperation({ summary: 'Submit ACH payment to JPMorgan' })
  async submitAchPayment(@Param('id') id: string) {
    await this.achService.submitToJpmorgan(id);
    return {
      success: true,
      message: 'ACH payment submitted successfully',
    };
  }
}
```

**Time:** 4 hours

---

#### **Task 3.3: Create ACH Webhook Controller**
**File:** `nestjs-backend/src/ach/controllers/ach-webhook.controller.ts`

**Implementation:**
```typescript
import { Controller, Post, Body, Logger } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';

@ApiTags('ACH Webhooks')
@Controller('api/webhooks/ach')
export class AchWebhookController {
  private readonly logger = new Logger(AchWebhookController.name);

  @Post()
  @ApiOperation({ summary: 'Receive ACH status webhook from JPMorgan' })
  async handleAchWebhook(@Body() payload: any) {
    this.logger.log('Received ACH webhook', JSON.stringify(payload));

    // TODO: Process webhook payload
    // - Update payment status
    // - Record payment event
    // - Send notifications

    return {
      success: true,
      message: 'Webhook received',
    };
  }
}
```

**Time:** 2 hours

---

#### **Task 3.4: Create ACH Response DTO**
**File:** `nestjs-backend/src/ach/dtos/ach-response.dto.ts`

**Implementation:**
```typescript
import { ApiProperty } from '@nestjs/swagger';

export class AchResponseDto {
  @ApiProperty()
  paymentId: string;

  @ApiProperty()
  achPaymentId: string;

  @ApiProperty()
  status: string;

  @ApiProperty()
  amount: number;

  @ApiProperty()
  currency: string;

  @ApiProperty()
  secCode: string;

  @ApiProperty()
  transactionType: string;

  @ApiProperty()
  receiverName: string;

  @ApiProperty()
  effectiveDate?: string;

  @ApiProperty()
  sameDayAch: boolean;

  @ApiProperty()
  createdAt: Date;
}
```

**Time:** 2 hours

---

### **Day 5: ACH Module Integration & Testing (8 hours)**

#### **Task 5.1: Create ACH Approval Guard**
**File:** `nestjs-backend/src/ach/guards/ach-approval.guard.ts`

**Implementation:**
```typescript
import { Injectable, CanActivate, ExecutionContext, ForbiddenException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Payment } from '../../payments-core/entities/payment.entity';
import { PaymentStatus } from '../../payments-core/enums/payment-status.enum';

@Injectable()
export class AchApprovalGuard implements CanActivate {
  constructor(
    @InjectRepository(Payment)
    private paymentRepository: Repository<Payment>,
  ) {}

  async canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const paymentId = request.params.id;

    if (!paymentId) {
      throw new ForbiddenException('Payment ID is required');
    }

    const payment = await this.paymentRepository.findOne({
      where: { id: paymentId },
    });

    if (!payment) {
      throw new ForbiddenException('Payment not found');
    }

    // Check if payment can be submitted
    if (payment.status !== PaymentStatus.APPROVED) {
      throw new ForbiddenException('Payment must be approved before submission');
    }

    return true;
  }
}
```

**Time:** 2 hours

---

#### **Task 5.2: Create ACH Module**
**File:** `nestjs-backend/src/ach/ach.module.ts`

**Implementation:**
```typescript
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { HttpModule } from '@nestjs/axios';
import { AchPayment } from './entities/ach-payment.entity';
import { Payment } from '../payments-core/entities/payment.entity';
import { AchService } from './services/ach.service';
import { AchValidationService } from './services/ach-validation.service';
import { NachaGeneratorService } from './services/nacha-generator.service';
import { AchJpmorganClient } from './services/ach-jpmorgan.client';
import { AchController } from './controllers/ach.controller';
import { AchWebhookController } from './controllers/ach-webhook.controller';
import { AchApprovalGuard } from './guards/ach-approval.guard';
import { PaymentsCoreModule } from '../payments-core/payments-core.module';
import { JpmorganModule } from '../connectors/jpmorgan/jpmorgan.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([AchPayment, Payment]),
    HttpModule,
    PaymentsCoreModule,
    JpmorganModule,
  ],
  providers: [
    AchService,
    AchValidationService,
    NachaGeneratorService,
    AchJpmorganClient,
    AchApprovalGuard,
  ],
  controllers: [
    AchController,
    AchWebhookController,
  ],
  exports: [AchService],
})
export class AchModule {}
```

**Time:** 2 hours

---

#### **Task 5.3: Update App Module**
**File:** `nestjs-backend/src/app.module.ts`

Add ACH module import:
```typescript
import { AchModule } from './ach/ach.module';

@Module({
  imports: [
    // ... existing imports
    AchModule,
  ],
})
export class AppModule {}
```

**Time:** 1 hour

---

#### **Task 5.4: Write Unit Tests**
**File:** `nestjs-backend/src/ach/services/ach.service.spec.ts`

**Implementation:**
```typescript
import { Test, TestingModule } from '@nestjs/testing';
import { AchService } from './ach.service';
import { getRepositoryToken } from '@nestjs/typeorm';
import { AchPayment } from '../entities/ach-payment.entity';
import { Payment } from '../../payments-core/entities/payment.entity';

describe('AchService', () => {
  let service: AchService;

  const mockAchPaymentRepository = {
    create: jest.fn(),
    save: jest.fn(),
    findOne: jest.fn(),
  };

  const mockPaymentRepository = {
    create: jest.fn(),
    save: jest.fn(),
    findOne: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AchService,
        {
          provide: getRepositoryToken(AchPayment),
          useValue: mockAchPaymentRepository,
        },
        {
          provide: getRepositoryToken(Payment),
          useValue: mockPaymentRepository,
        },
        // Add other mocked dependencies
      ],
    }).compile();

    service = module.get<AchService>(AchService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should create ACH payment', async () => {
    // Add test implementation
  });

  it('should validate routing number', async () => {
    // Add test implementation
  });
});
```

**Time:** 3 hours

---

## **WEEK 1 SUMMARY**

### **Completed Files (12/12):**
✅ `src/ach/entities/ach-payment.entity.ts` (already exists)  
✅ `src/ach/dtos/create-ach.dto.ts` (already exists)  
✅ `src/ach/dtos/ach-response.dto.ts`  
✅ `src/ach/services/ach-validation.service.ts`  
✅ `src/ach/services/nacha-generator.service.ts`  
✅ `src/ach/services/ach-jpmorgan.client.ts`  
✅ `src/ach/services/ach.service.ts`  
✅ `src/ach/controllers/ach.controller.ts`  
✅ `src/ach/controllers/ach-webhook.controller.ts`  
✅ `src/ach/guards/ach-approval.guard.ts`  
✅ `src/ach/ach.module.ts`  
✅ `src/app.module.ts` (updated)

### **Testing:**
✅ Unit tests for all services  
✅ Integration tests for controllers  
✅ End-to-end ACH payment flow test

### **Total Time:** 40 hours

---

## 📅 WEEK 2 & 3 PREVIEW

### **Week 2: Wire & RTP Modules (40 hours)**
- Day 1-2: Wire Transfer Module (16 hours)
- Day 3-4: RTP Payment Module (16 hours)
- Day 5: Approval Workflow System (8 hours)

### **Week 3: Dashboard & Testing (16 hours)**
- Day 1-2: Enhanced Live Transaction Dashboard (8 hours)
- Day 3-4: Integration Testing (6 hours)
- Day 5: Documentation & Handoff (2 hours)

---

## 🚀 IMMEDIATE NEXT STEPS

### **This Week (Week 1):**

1. **Start Day 1 Tasks** (Today)
   - Create ACH validation service
   - Create NACHA generator service

2. **Continue Day 2 Tasks** (Tomorrow)
   - Create ACH JP Morgan client
   - Test API integration

3. **Complete Week 1** (Days 3-5)
   - Implement ACH service & controllers
   - Create ACH module
   - Write tests

### **Parallel Track: JP Morgan Onboarding**
While building sandbox system, start production onboarding:
- Contact JP Morgan Treasury Services
- Request production API access
- Gather business documentation

---

## 📊 SUCCESS CRITERIA

### **Week 1 Complete When:**
- [ ] All 12 ACH files created
- [ ] ACH module compiles without errors
- [ ] All unit tests passing
- [ ] ACH payment can be created via API
- [ ] ACH payment can be submitte
